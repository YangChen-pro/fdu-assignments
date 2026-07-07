import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_video import TrackPostProcessor, class_family


class TrackVideoLogicTest(unittest.TestCase):
    def test_reconnects_new_raw_id_after_short_occlusion(self):
        processor = TrackPostProcessor(
            line_px=640,
            use_vertical=True,
            reconnect_window=90,
            reconnect_distance_px=180,
        )

        processor.begin_frame()
        first = processor.update(
            raw_track_id=10,
            class_name="car",
            cx=500,
            cy=420,
            width=120,
            height=70,
            frame_idx=1,
        )

        processor.begin_frame()
        processor.update(
            raw_track_id=10,
            class_name="car",
            cx=560,
            cy=420,
            width=120,
            height=70,
            frame_idx=11,
        )

        processor.begin_frame()
        reconnected = processor.update(
            raw_track_id=77,
            class_name="taxi",
            cx=680,
            cy=420,
            width=118,
            height=72,
            frame_idx=31,
        )

        self.assertEqual(first.display_id, reconnected.display_id)
        self.assertTrue(reconnected.reconnected)
        self.assertEqual(processor.count, 1)
        self.assertEqual(processor.crossed_display_ids, {first.display_id})
        self.assertEqual(processor.predicted_crossings, 0)

    def test_does_not_reconnect_distant_new_track(self):
        processor = TrackPostProcessor(
            line_px=640,
            use_vertical=True,
            reconnect_window=90,
            reconnect_distance_px=120,
        )

        processor.begin_frame()
        first = processor.update(
            raw_track_id=10,
            class_name="car",
            cx=500,
            cy=420,
            width=120,
            height=70,
            frame_idx=1,
        )

        processor.begin_frame()
        second = processor.update(
            raw_track_id=88,
            class_name="car",
            cx=1050,
            cy=420,
            width=120,
            height=70,
            frame_idx=20,
        )

        self.assertNotEqual(second.display_id, first.display_id)
        self.assertFalse(second.reconnected)

    def test_does_not_reuse_display_id_when_previous_raw_id_is_in_same_frame(self):
        processor = TrackPostProcessor(
            line_px=640,
            use_vertical=True,
            reconnect_window=90,
            reconnect_distance_px=180,
        )

        processor.begin_frame(raw_track_ids=[10])
        first = processor.update(
            raw_track_id=10,
            class_name="car",
            cx=500,
            cy=420,
            width=120,
            height=70,
            frame_idx=1,
        )

        processor.begin_frame(raw_track_ids=[10])
        processor.update(
            raw_track_id=10,
            class_name="car",
            cx=560,
            cy=420,
            width=120,
            height=70,
            frame_idx=11,
        )

        processor.begin_frame(raw_track_ids=[77, 10])
        new_track = processor.update(
            raw_track_id=77,
            class_name="taxi",
            cx=680,
            cy=420,
            width=118,
            height=72,
            frame_idx=31,
        )
        old_track = processor.update(
            raw_track_id=10,
            class_name="car",
            cx=620,
            cy=420,
            width=120,
            height=70,
            frame_idx=31,
        )

        self.assertEqual(old_track.display_id, first.display_id)
        self.assertNotEqual(new_track.display_id, first.display_id)
        self.assertFalse(new_track.reconnected)

    def test_predicts_crossing_while_track_is_temporarily_missing(self):
        processor = TrackPostProcessor(
            line_px=640,
            use_vertical=True,
            reconnect_window=90,
            reconnect_distance_px=180,
        )

        processor.begin_frame()
        first = processor.update(
            raw_track_id=10,
            class_name="car",
            cx=500,
            cy=420,
            width=120,
            height=70,
            frame_idx=1,
        )

        processor.begin_frame()
        processor.update(
            raw_track_id=10,
            class_name="car",
            cx=560,
            cy=420,
            width=120,
            height=70,
            frame_idx=11,
        )

        for frame_idx in range(12, 26):
            processor.begin_frame()
            processor.finish_frame(frame_idx)

        self.assertEqual(processor.count, 1)
        self.assertEqual(processor.crossed_display_ids, {first.display_id})
        self.assertEqual(processor.predicted_crossings, 1)

    def test_class_family_groups_common_vehicle_label_flips(self):
        self.assertEqual(class_family("car"), class_family("taxi"))
        self.assertEqual(class_family("taxi"), class_family("suv"))
        self.assertEqual(class_family("motorbike"), class_family("bicycle"))
        self.assertNotEqual(class_family("bus"), class_family("car"))


if __name__ == "__main__":
    unittest.main()
