from enum import Enum
from data_io.common import Color

class Status(Enum):
    Enter = 0           # frame_count <= 1
    Validating = 1      # frame_count in range[1:5]
    Classifying = 2     # frame_count > 5 & not counted
    Counted = 3         # frame_count > 5 & counted
    Exit = 4

class VehicleType(Enum):
    Unidentified = 0
    Class1 = 1
    Class2 = 2
    Class3 = 3
    Vehicle_Block = 4

class TravelingStatus(Enum):
    Normal = 0
    WrongWayDriving = 1
    Overspeeding = 2
    UnsafeLaneChange = 3

class Direction(Enum):
    Downstream = 1
    Upstream = 2

VehicleColor = [Color.White.value,
                Color.Green.value,
                Color.Yellow.value,
                Color.Blue.value,
                Color.Red.value]