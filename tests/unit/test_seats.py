"""Automatic seats in the student camera (main code/Main Code.py: find_seats, assign_students)."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("main_code_seats", ROOT / "main code" / "Main Code.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

W, H = 1280, 720
NO_OBJECTS = pd.DataFrame(columns=["frame", "label", "px", "py"])
AUTO = {"student_region": [0.0, 0.0, 1.0, 1.0], "seats": "auto", "seat_grid": [3, 4]}


def person(i, sx, sy, sw=80.0, shoulders=True, box=None):
    """One student detection: shoulder midpoint (sx, sy) and width sw in pixels; box defaults to around the shoulders."""
    x1, y1, x2, y2 = box or (sx - sw, sy - sw, sx + sw, sy + 1.5 * sw)
    row = {"camera": "students", "frame": i, "time_s": 2.0 * i, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
           "cx": (x1 + x2) / 2 / W, "cy": (y1 + y2) / 2 / H, "conf": 0.85, **mc.NO_CUES, "phone": False, "book": False,
           "role": "other", "seat": "", "candidates": 0, "img_w": W, "img_h": H}
    if shoulders:
        row.update(sx=sx, sy=sy, sw=sw, head_up=True)
    return row


def seat_people(rows, n, calib=AUTO):
    return mc.assign_students(pd.DataFrame(rows, columns=mc.PEOPLE_COLUMNS), NO_OBJECTS, dict(calib), n_frames=n)


class AutomaticSeatTests(unittest.TestCase):
    def test_classroom_like_the_real_one(self):
        """Six students in two rows, one of them restless; legs under a desk and a chair are each seen twice."""
        back, front = [(320, 204), (478, 208), (658, 202)], [(265, 369), (508, 363), (699, 330)]
        rows = []
        for i in range(50):
            for k, (x, y) in enumerate(back):
                rows.append(person(i, x + (20 * np.sin(i) if k == 0 else 0), y, sw=84))
            for x, y in front:
                rows.append(person(i, x, y, sw=110))
            if i in (8, 30):
                rows.append(person(i, 680, 650, shoulders=False, box=(627, 592, 737, 707)))   # legs under a desk
            if i in (3, 23):
                rows.append(person(i, 1120, 650, shoulders=False, box=(1005, 574, 1236, 720)))  # a chair
        people = seat_people(rows, 50)
        students = people[people["role"] == "student"]
        self.assertEqual(sorted(students["seat"].unique()), [f"seat_{k:02d}" for k in range(1, 7)])
        self.assertEqual(int((people["role"] == "other").sum()), 4)                  # the four ghosts are left out
        self.assertEqual(int(students.groupby(["frame", "seat"]).size().max()), 1)   # never two people in one seat
        # numbered row by row from the top left: the back row (higher in the image) is seats 1-3
        first = students[students["frame"] == 0].sort_values("seat")
        self.assertEqual(list(first["sx"].round()), [320, 478, 658, 265, 508, 699])

    def test_neighbours_close_together_stay_apart(self):
        rows = [person(i, x, 300) for i in range(20) for x in (400, 520)]          # 1.5 shoulder widths apart
        people = seat_people(rows, 20)
        self.assertEqual(people.groupby("seat")["frame"].nunique().to_dict(), {"seat_01": 20, "seat_02": 20})

    def test_a_student_who_shifts_keeps_one_seat(self):
        rows = [person(i, 400 if i < 10 else 460, 300) for i in range(20)]          # moves 0.75 shoulder widths once
        self.assertEqual(set(seat_people(rows, 20)["seat"]), {"seat_01"})

    def test_a_student_who_stands_up_briefly_stays_in_their_seat(self):
        rows = [person(i, 400, 200 if 20 <= i < 23 else 300) for i in range(40)]   # shoulders 1.25 widths higher
        people = seat_people(rows, 40)
        self.assertEqual(set(people["seat"]), {"seat_01"})
        self.assertTrue((people["role"] == "student").all())

    def test_hidden_shoulders_count_only_on_a_free_seat_they_cover(self):
        rows = [person(i, 400, 300) for i in range(20)]
        rows.append(person(20, 0, 0, shoulders=False, box=(330, 220, 470, 420)))   # same spot, turned away, seat free
        rows.append(person(19, 0, 0, shoulders=False, box=(330, 220, 470, 420)))   # seat already taken in this sample
        people = seat_people(rows, 21)
        self.assertEqual(list(people["role"].iloc[-2:]), ["student", "other"])
        self.assertEqual(people["seat"].iloc[-2], "seat_01")

    def test_a_place_sat_in_only_briefly_is_no_seat(self):
        rows = [person(i, 400, 300) for i in range(40)] + [person(i, 900, 300) for i in range(3)]   # 3 of 40 samples
        people = seat_people(rows, 40)
        self.assertEqual(set(people.loc[people["role"] == "student", "seat"]), {"seat_01"})

    def test_grid_mode_is_still_available(self):
        rows = [person(0, 100, 100), person(0, 1200, 650, shoulders=False)]
        people = seat_people(rows, 1, {"student_region": [0.0, 0.0, 1.0, 1.0], "seats": "grid", "seat_grid": [3, 4]})
        self.assertEqual(list(people["seat"]), ["seat_01", "seat_12"])
        self.assertTrue((people["role"] == "student").all())


if __name__ == "__main__":
    unittest.main()
