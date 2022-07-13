from math import atan2, degrees
import re
from datetime import datetime

def angle_between(p1, p2, p3):
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

# Get current time (yyyy_mm_dd_hh_mm_ss_ms.jpg)
def get_current_frame_name():
    return re.sub(r"[^0-9]", '_', str(datetime.now())) + '.jpg'