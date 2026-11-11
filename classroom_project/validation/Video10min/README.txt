How to fill in the sheets (about 20-30 minutes)

review_events.csv  - one row per detected event. Open the image named in the 'image' column.
  is_real : TRUE if the event really happened and was attributed to the right person (a hand raise by a STUDENT,
            a phone really in use), FALSE if not. Use 'note' for anything odd (e.g. "this is the instructor").

review_frames.csv  - a random sample of frames. Open the image named in the 'image' column (boxes show what the
  program thought). Fill in:
  instructor_visible    : TRUE / FALSE - is the instructor visible in the frame at all?
  instructor_box_correct: TRUE / FALSE - is the box labelled 'instructor' really the instructor? (leave blank if none)
  instructor_posture    : sitting / standing / cannot_tell
  hand_raise_seats      : the seat labels (as printed on the image) of students whose hand is raised,
                          separated by ';' (e.g. seat_03;seat_08), or 'none'

Leave a cell blank if you cannot tell. Do not edit the other columns. Label what you SEE, not what the boxes say.
