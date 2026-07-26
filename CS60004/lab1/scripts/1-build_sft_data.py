import os
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocess_utils import (
    TeacherModel,
    read_jsonl,
    write_jsonl,
    append_jsonl,
    convert_one_sample,
    process_one_sample,
)

def process_dataset(input_path, output_path, plan='A', max_samples=None, teacher=None, num_workers=1):
    raw_samples = read_jsonl(input_path)
    if max_samples is not None:
        raw_samples = raw_samples[:max_samples]

    if plan in ['B', 'C']: # B/C方案边处理边写，需要先清空输出文件
        save_dir = os.path.dirname(output_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        open(output_path, 'w', encoding='utf-8').close()

    if plan == 'A' or num_workers <= 1:
        new_samples = []
        for i, sample in enumerate(tqdm(raw_samples, desc=f'Plan {plan}', ncols=80)):
            try:
                new_sample = convert_one_sample(sample, plan=plan, teacher=teacher)
                if plan == 'A': # A方案快，可以最后一次写
                    new_samples.append(new_sample)
                else: # B/C方案慢，边处理边写
                    append_jsonl(output_path, new_sample)
            except Exception as e:
                print(f'[Skip] sample {i}: {e}')

        if plan == 'A':
            write_jsonl(output_path, new_samples)

    else:
        pbar = tqdm(total=len(raw_samples), desc=f'Plan {plan}', ncols=80)

        for start in range(0, len(raw_samples), num_workers):
            end = min(start + num_workers, len(raw_samples))
            batch_samples = raw_samples[start:end]

            batch_results = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for offset, sample in enumerate(batch_samples):
                    i = start + offset
                    future = executor.submit(
                        process_one_sample,
                        i,
                        sample,
                        plan,
                        teacher
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    i, new_sample, err = future.result()
                    batch_results.append((i, new_sample, err))
                    pbar.update(1)

            batch_results.sort(key=lambda x: x[0])

            for i, new_sample, err in batch_results:
                if err is not None:
                    print(f'[Skip] sample {i}: {err}')
                    continue
                append_jsonl(output_path, new_sample)

        pbar.close()

    print(f'Input file   : {input_path}')
    print(f'Output file  : {output_path}')
    print(f'Plan         : {plan}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        default='datasets/provide_to_students/train.jsonl'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='datasets/task1/plan_a_train.jsonl'
    )
    parser.add_argument(
        '--plan',
        type=str,
        default='A',
        choices=['A', 'B', 'C']
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    teacher = None
    if args.plan in ['B', 'C']:
        teacher = TeacherModel(
            model_name='qwen-3.5-9b',
            base_url='http://localhost:12345/v1'
        )
    process_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        plan=args.plan,
        max_samples=args.max_samples,
        teacher=teacher,
        num_workers=args.num_workers
    )

# python scripts/1-build_sft_data.py --input_path datasets/provide_to_students/train.jsonl --output_path datasets/task1/plan_c_train.jsonl --plan C --num_workers 16 --max_samples 20