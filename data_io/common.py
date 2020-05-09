from enum import Enum
import colorama

class Color(Enum):
    Red = (34, 68, 228)
    LightRed = (101, 74, 254)
    Green = (97, 204, 66)
    LightGreen = (0, 255, 127)
    ArmyGreen = (75, 170, 30)
    Blue = (225, 151, 60)
    DarkBlue = (255, 146, 1)
    Emerald = (152, 188, 53)
    White = (241, 240, 237)
    LightGray = (96, 71, 51)
    Gray = (50, 50, 50)
    Yellow = (1, 210, 254)

class ConsoleColor(Enum):
    Black   = colorama.Fore.BLACK
    Red     = colorama.Fore.RED
    Green   = colorama.Fore.GREEN
    Yellow  = colorama.Fore.YELLOW
    Blue    = colorama.Fore.BLUE
    Magenta = colorama.Fore.MAGENTA
    Cyan    = colorama.Fore.CYAN
    White   = colorama.Fore.WHITE



