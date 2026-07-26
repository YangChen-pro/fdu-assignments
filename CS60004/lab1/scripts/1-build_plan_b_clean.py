import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from preprocess_utils import (
    JudgeOutputFormatError,
    TeacherModel,
    TeacherEmptyContentError,
    check_plan_b_clean_hard_rules,
    judge_plan_b_clean_output,
    read_jsonl,
    regenerate_plan_b_clean_output,
    write_jsonl,
)


def decide_from_judge_result(judge_result, keep_threshold=8, drop_threshold=4):
    """Map teacher-judge scores to keep/regen/drop."""
    total_score = judge_result['total_score']
    decision = judge_result['decision']

    if decision == 'drop' or total_score <= drop_threshold:
        return 'drop'
    if decision == 'keep' and total_score >= keep_threshold:
        return 'keep'
    return 'regen'


def build_clean_sample(sample, output_str):
    """Build a cleaned sample while preserving instruction and input."""
    return {
        'instruction': sample['instruction'],
        'input': sample['input'],
        'output': output_str,
    }


def judge_plan_b_clean_output_with_retry(
    sample,
    output_str,
    teacher,
    max_tokens=4096,
    retry_max_tokens_list=None
):
    """Retry judge calls on truncation or one-missing-subscore errors."""
    token_schedule = [max_tokens]
    if retry_max_tokens_list is None:
        retry_max_tokens_list = [8192, 12288]
    for item in retry_max_tokens_list:
        if item > token_schedule[-1]:
            token_schedule.append(item)

    token_index = 0
    retried_format_error_types = set()
    retry_events = []

    while True:
        try:
            judge_result = judge_plan_b_clean_output(
                sample,
                output_str,
                teacher,
                max_tokens=token_schedule[token_index],
            )
            return judge_result, retry_events
        except TeacherEmptyContentError as e:
            if e.finish_reason == 'length' and token_index + 1 < len(token_schedule):
                token_index += 1
                retry_events.append(f'length_retry_to_{token_schedule[token_index]}')
                continue
            raise
        except JudgeOutputFormatError as e:
            if e.retryable and e.error_type not in retried_format_error_types:
                retried_format_error_types.add(e.error_type)
                if e.error_type == 'missing_subscore':
                    retry_events.append(
                        'missing_subscore_retry:' + ','.join(e.missing_subscores)
                    )
                elif e.error_type == 'json_decode':
                    retry_events.append('json_decode_retry')
                else:
                    retry_events.append(f'{e.error_type}_retry')
                continue
            raise


def process_plan_b_clean_sample(
    i,
    sample,
    teacher,
    keep_threshold=8,
    drop_threshold=4,
    max_regen_attempts=2,
    judge_max_tokens=4096,
    regen_max_tokens=4096
):
    """Process one Plan B sample into keep/regen/drop."""
    audit = {
        'index': i,
        'original_hard_pass': False,
        'original_hard_reasons': [],
        'original_judge': None,
        'original_judge_retry_events': [],
        'regen_attempts': [],
        'final_status': None,
        'final_reason': '',
    }

    try:
        original_output = sample['output']
        hard_pass, hard_reasons = check_plan_b_clean_hard_rules(original_output)
        audit['original_hard_pass'] = hard_pass
        audit['original_hard_reasons'] = hard_reasons

        if hard_pass:
            judge_result, retry_events = judge_plan_b_clean_output_with_retry(
                sample,
                original_output,
                teacher,
                max_tokens=judge_max_tokens,
            )
            audit['original_judge'] = judge_result
            audit['original_judge_retry_events'] = retry_events
            action = decide_from_judge_result(
                judge_result,
                keep_threshold=keep_threshold,
                drop_threshold=drop_threshold,
            )
            if action == 'keep':
                audit['final_status'] = 'keep_original'
                audit['final_reason'] = judge_result['reason']
                return i, sample, audit, None
            if action == 'drop':
                audit['final_status'] = 'drop_original'
                audit['final_reason'] = judge_result['reason']
                return i, None, audit, None

        for attempt_id in range(1, max_regen_attempts + 1):
            regenerated_output = regenerate_plan_b_clean_output(
                sample,
                teacher,
                max_tokens=regen_max_tokens,
            )
            regen_hard_pass, regen_hard_reasons = check_plan_b_clean_hard_rules(regenerated_output)
            attempt_info = {
                'attempt_id': attempt_id,
                'hard_pass': regen_hard_pass,
                'hard_reasons': regen_hard_reasons,
                'judge': None,
                'judge_retry_events': [],
            }

            if regen_hard_pass:
                regen_judge, retry_events = judge_plan_b_clean_output_with_retry(
                    sample,
                    regenerated_output,
                    teacher,
                    max_tokens=judge_max_tokens,
                )
                attempt_info['judge'] = regen_judge
                attempt_info['judge_retry_events'] = retry_events
                action = decide_from_judge_result(
                    regen_judge,
                    keep_threshold=keep_threshold,
                    drop_threshold=drop_threshold,
                )

                if action == 'keep':
                    audit['regen_attempts'].append(attempt_info)
                    audit['final_status'] = 'keep_regen'
                    audit['final_reason'] = regen_judge['reason']
                    return i, build_clean_sample(sample, regenerated_output), audit, None
                if action == 'drop':
                    audit['regen_attempts'].append(attempt_info)
                    audit['final_status'] = 'drop_regen'
                    audit['final_reason'] = regen_judge['reason']
                    return i, None, audit, None

            audit['regen_attempts'].append(attempt_info)

        audit['final_status'] = 'drop_regen'
        if hard_pass and audit['original_judge'] is not None:
            audit['final_reason'] = audit['original_judge']['reason']
        elif hard_reasons:
            audit['final_reason'] = ', '.join(hard_reasons)
        else:
            audit['final_reason'] = 'regen_failed'
        return i, None, audit, None

    except Exception as e:
        audit['final_status'] = 'error'
        audit['final_reason'] = str(e)
        return i, None, audit, e


def collect_results(
    samples,
    teacher,
    keep_threshold=8,
    drop_threshold=4,
    max_regen_attempts=2,
    judge_max_tokens=4096,
    regen_max_tokens=4096,
    num_workers=1
):
    """Run Plan B cleaning over a batch of samples."""
    cleaned_samples = []
    audits = []

    if num_workers <= 1:
        for i, sample in enumerate(tqdm(samples, desc='Plan B Clean', ncols=80)):
            _, cleaned_sample, audit, err = process_plan_b_clean_sample(
                i,
                sample,
                teacher,
                keep_threshold=keep_threshold,
                drop_threshold=drop_threshold,
                max_regen_attempts=max_regen_attempts,
                judge_max_tokens=judge_max_tokens,
                regen_max_tokens=regen_max_tokens,
            )
            audits.append(audit)
            if err is not None:
                print(f'[Skip] sample {i}: {err}')
                continue
            if cleaned_sample is not None:
                cleaned_samples.append((i, cleaned_sample))
    else:
        pbar = tqdm(total=len(samples), desc='Plan B Clean', ncols=80)
        indexed_results = []

        for start in range(0, len(samples), num_workers):
            end = min(start + num_workers, len(samples))
            batch_samples = samples[start:end]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for offset, sample in enumerate(batch_samples):
                    i = start + offset
                    future = executor.submit(
                        process_plan_b_clean_sample,
                        i,
                        sample,
                        teacher,
                        keep_threshold,
                        drop_threshold,
                        max_regen_attempts,
                        judge_max_tokens,
                        regen_max_tokens,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    indexed_results.append(future.result())
                    pbar.update(1)

        pbar.close()
        indexed_results.sort(key=lambda x: x[0])

        for i, cleaned_sample, audit, err in indexed_results:
            audits.append(audit)
            if err is not None:
                print(f'[Skip] sample {i}: {err}')
                continue
            if cleaned_sample is not None:
                cleaned_samples.append((i, cleaned_sample))

    cleaned_samples.sort(key=lambda x: x[0])
    return [sample for _, sample in cleaned_samples], audits


def select_samples(samples, dry_run=False, dry_run_size=50):
    """Select full data or a deterministic dry-run subset."""
    if not dry_run:
        return samples
    return samples[:min(dry_run_size, len(samples))]


def build_report_text(
    audits,
    total_input,
    keep_threshold=8,
    drop_threshold=4,
    max_regen_attempts=2,
    dry_run=False
):
    """Build the plain-text report for Plan B cleaning."""
    final_counter = Counter(audit['final_status'] for audit in audits)
    hard_reason_counter = Counter()
    judge_decision_counter = Counter()
    regen_judge_counter = Counter()
    regen_requested = 0

    keep_examples = []
    regen_examples = []
    drop_examples = []
    error_examples = []

    for audit in audits:
        for reason in audit['original_hard_reasons']:
            hard_reason_counter[reason] += 1

        if audit['original_judge'] is not None:
            judge_decision_counter[audit['original_judge']['decision']] += 1

        if audit['regen_attempts']:
            regen_requested += 1
            for attempt in audit['regen_attempts']:
                if attempt['judge'] is not None:
                    regen_judge_counter[attempt['judge']['decision']] += 1

        if audit['final_status'] in ['keep_original', 'keep_regen'] and len(keep_examples) < 5:
            keep_examples.append(f'{audit["index"]}: {audit["final_reason"]}')
        if audit['regen_attempts'] and len(regen_examples) < 5:
            latest_attempt = audit['regen_attempts'][-1]
            regen_examples.append(
                f'{audit["index"]}: hard_pass={latest_attempt["hard_pass"]}, '
                f'hard_reasons={latest_attempt["hard_reasons"]}'
            )
        if audit['final_status'] in ['drop_original', 'drop_regen'] and len(drop_examples) < 5:
            drop_examples.append(f'{audit["index"]}: {audit["final_reason"]}')
        if audit['final_status'] == 'error' and len(error_examples) < 5:
            error_examples.append(f'{audit["index"]}: {audit["final_reason"]}')

    lines = [
        'Plan B Clean Report',
        '===================',
        '',
        f'dry_run            : {dry_run}',
        f'input_samples      : {total_input}',
        f'processed_samples  : {len(audits)}',
        f'keep_threshold     : {keep_threshold}',
        f'drop_threshold     : {drop_threshold}',
        f'max_regen_attempts : {max_regen_attempts}',
        '',
        'Final counts:',
        f'- keep_total    : {final_counter["keep_original"] + final_counter["keep_regen"]}',
        f'- keep_original : {final_counter["keep_original"]}',
        f'- keep_regen    : {final_counter["keep_regen"]}',
        f'- regen_total   : {regen_requested}',
        f'- drop_total    : {final_counter["drop_original"] + final_counter["drop_regen"]}',
        f'- drop_original : {final_counter["drop_original"]}',
        f'- drop_regen    : {final_counter["drop_regen"]}',
        f'- error         : {final_counter["error"]}',
        f'- final_b_clean : {final_counter["keep_original"] + final_counter["keep_regen"]}',
        '',
        'Original hard-rule reasons:',
    ]

    if hard_reason_counter:
        for key in sorted(hard_reason_counter):
            lines.append(f'- {key}: {hard_reason_counter[key]}')
    else:
        lines.append('- none')

    lines.extend([
        '',
        'Original judge decisions:',
    ])

    if judge_decision_counter:
        for key in sorted(judge_decision_counter):
            lines.append(f'- {key}: {judge_decision_counter[key]}')
    else:
        lines.append('- none')

    lines.extend([
        '',
        'Regen judge decisions:',
    ])

    if regen_judge_counter:
        for key in sorted(regen_judge_counter):
            lines.append(f'- {key}: {regen_judge_counter[key]}')
    else:
        lines.append('- none')

    lines.extend([
        '',
        'Keep examples:',
    ])
    lines.extend([f'- {item}' for item in keep_examples] or ['- none'])

    lines.extend([
        '',
        'Regen examples:',
    ])
    lines.extend([f'- {item}' for item in regen_examples] or ['- none'])

    lines.extend([
        '',
        'Drop examples:',
    ])
    lines.extend([f'- {item}' for item in drop_examples] or ['- none'])

    lines.extend([
        '',
        'Error examples:',
    ])
    lines.extend([f'- {item}' for item in error_examples] or ['- none'])

    return '\n'.join(lines) + '\n'


def write_report(path, text):
    """Write the plain-text report."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        default='datasets/task1/plan_b_train.jsonl'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='datasets/task1/plan_b_clean_train.jsonl'
    )
    parser.add_argument(
        '--report_path',
        type=str,
        default='datasets/task1/plan_b_clean_report.txt'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1
    )
    parser.add_argument(
        '--dry_run',
        action='store_true'
    )
    parser.add_argument(
        '--dry_run_size',
        type=int,
        default=50
    )
    parser.add_argument(
        '--keep_threshold',
        type=int,
        default=8
    )
    parser.add_argument(
        '--drop_threshold',
        type=int,
        default=4
    )
    parser.add_argument(
        '--max_regen_attempts',
        type=int,
        default=2
    )
    parser.add_argument(
        '--judge_max_tokens',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--regen_max_tokens',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--teacher_model_name',
        type=str,
        default='qwen-3.5-9b'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default='http://localhost:12345/v1'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    teacher = TeacherModel(
        model_name=args.teacher_model_name,
        base_url=args.base_url
    )

    samples = read_jsonl(args.input_path)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    samples = select_samples(
        samples,
        dry_run=args.dry_run,
        dry_run_size=args.dry_run_size,
    )

    cleaned_samples, audits = collect_results(
        samples=samples,
        teacher=teacher,
        keep_threshold=args.keep_threshold,
        drop_threshold=args.drop_threshold,
        max_regen_attempts=args.max_regen_attempts,
        judge_max_tokens=args.judge_max_tokens,
        regen_max_tokens=args.regen_max_tokens,
        num_workers=args.num_workers,
    )

    report_text = build_report_text(
        audits=audits,
        total_input=len(samples),
        keep_threshold=args.keep_threshold,
        drop_threshold=args.drop_threshold,
        max_regen_attempts=args.max_regen_attempts,
        dry_run=args.dry_run,
    )

    write_jsonl(args.output_path, cleaned_samples)
    write_report(args.report_path, report_text)

    print(f'Input file   : {args.input_path}')
    print(f'Output file  : {args.output_path}')
    print(f'Report file  : {args.report_path}')
    print(f'Dry run      : {args.dry_run}')
    print(f'Input count  : {len(samples)}')
    print(f'Output count : {len(cleaned_samples)}')


if __name__ == '__main__':
    main()


# python scripts/1-build_plan_b_clean.py --dry_run --dry_run_size 50 --num_workers 8
# python scripts/1-build_plan_b_clean.py --num_workers 16
