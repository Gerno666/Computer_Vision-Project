import numpy as np
from staff_detection import find_line_equation
import math

SCALE = 'C', 'D', 'E', 'F', 'G', 'A', 'H'


def does_belong_to_staff(staff, sign):
    staff_height = staff[4][1] - staff[0][1]
    x_ul = staff[0][0]
    y_ul = staff[0][1] - staff_height / 2
    x_ur = staff[0][2]
    y_ur = staff[0][3] - staff_height / 2
    x_ll = staff[4][0]
    y_ll = staff[4][1] + staff_height / 2
    x_lr = staff[4][2]
    y_lr = staff[4][3] + staff_height / 2
    sign_left = sign[0][0]
    sign_right = sign[1][0]
    sign_top = sign[0][1]
    sign_bottom = sign[1][1]
    if sign_right < min(x_ul, x_ll) or sign_left > max(x_ur, x_lr) or sign_top > max(y_ll, y_lr) or sign_bottom < min(y_ul, y_ur):
        return False
    else:
        return True


'''
Returns a number of line in staff on which the sign is.
Lines are numbered from down to up, from 0 to 4.
Fields will have half number, e.g. 0.5 means the first (lowest) field.
Other lines/fields (above or under the staff are calculated in the same way.
'''


def what_line(x, y, lines_data, distance_between_lines):
    close_to_line = distance_between_lines/4
    top_y = lines_data[0][0]*x+lines_data[0][1]
    if abs(top_y - y) < close_to_line:
        return 4
    elif y < top_y:
        added_lines = 1
        found_y = top_y-distance_between_lines*added_lines
        if abs(found_y-y) < close_to_line:
            return 5
        while found_y > y:
            added_lines += 1
            found_y = top_y - distance_between_lines * added_lines
            if abs(found_y - y) < close_to_line:
                return 4+added_lines
        return 3.5+added_lines
    else:
        bottom_y = None
        for i in range(1, 5):
            current_y = lines_data[i][0]*x+lines_data[i][1]
            if abs(current_y-y) < close_to_line:
                return 4-i
            elif y < current_y:
                return 4.5-i
            if i == 4:
                bottom_y = current_y
        added_lines = 1
        found_y = bottom_y + distance_between_lines * added_lines
        if abs(found_y - y) < close_to_line:
            return -1
        while found_y < y:
            added_lines += 1
            found_y = bottom_y + distance_between_lines * added_lines
            if abs(found_y - y) < close_to_line:
                return 0 - added_lines
        return 0.5 - added_lines


def sign_center(sign):
    x = np.mean([sign[0][0], sign[1][0]])
    y = np.mean([sign[0][1], sign[1][1]])
    return x, y


def find_tone(clef, line_number):
    tones_from_first_line = int(line_number / 0.5)
    if clef == "treble_clef":
        tones_from_G = tones_from_first_line-2
        tones_from_C = tones_from_first_line+2
        octave = math.floor(tones_from_C / 7)
        return SCALE[(4+tones_from_G) % 7], octave
    else:
        tones_from_F = tones_from_first_line-6
        tones_from_C = tones_from_first_line-3
        octave = math.floor(tones_from_C / 7) - 1
        return SCALE[(3+tones_from_F) % 7], octave


def create_sounds_sequence(staff, signs):

    mean_distance_between_lines = 0
    for i in range(1, 5):
        mean_distance_between_lines += staff[i][1] - staff[i-1][1]
    mean_distance_between_lines /= 4

    sounds = []
    repetition = []
    is_repetition = False

    # 1. find this staff's signs
    '''sequence = []
    to_delete = []
    for sign in signs:
        if does_belong_to_staff(staff, sign):
            sequence.append(tuple(sign))
            to_delete.append(sign)

    for sign in to_delete:
        signs.remove(sign)
    to_delete.clear()'''
    # No filtering sounds
    sequence = []
    for sign in signs:
        sequence.append(sign)

    sequence = sorted(sequence, key=lambda x: x[0], reverse=False)

    # 2. interpretation

    # lines data (equations parameters)
    lines_data = []
    for line in staff:
        lines_data.append(find_line_equation(line))

    # signs
    clef = None
    clef_signs = []
    are_sharps = None
    locally_deleted_signs = []
    local_sharps = []
    local_flats = []

    previous_sign = None
    for sign in sequence:
        if sign[2] == "treble_clef":
            clef = "treble_clef"
        elif sign[2] == "bass_clef":
            clef = "bass_clef"
        elif sign[2] == "sharp":
            if clef is None:
                print("ERROR: No clef")
                return None
            x, y = sign_center(sign)
            y -= int((y-sign[0][1])/3)
            line_number = what_line(x, y, lines_data, mean_distance_between_lines)
            print("sharp on ", line_number)
            if previous_sign[2] == "treble_clef" or previous_sign[2] == "bass_clef" or previous_sign[2] == "sharp":
                if len(clef_signs) == 0:
                    if (clef == "treble_clef" and line_number == 4) or (clef == "bass_clef" and line_number == 3):
                        clef_signs.append('F')
                else:
                    expected = (SCALE.index(clef_signs[-1])-3) % 7
                    clef_signs.append(SCALE[expected])  # should we check correctness? or assume notes are correct
                are_sharps = True
            else:
                local_sharps.append(find_tone(clef, line_number)[0])
        elif sign[2] == "flat":
            if clef is None:
                print("ERROR: No clef")
                return None
            x, y = sign_center(sign)
            line_number = what_line(x, y, lines_data, mean_distance_between_lines)
            print("flat on ", line_number)
            if previous_sign[2] == "treble_clef" or previous_sign[2] == "bass_clef" or previous_sign[2] == "flat":
                if len(clef_signs) == 0:
                    if (clef == "treble_clef" and line_number == 2) or (clef == "bass_clef" and line_number == 1):
                        clef_signs.append('H')
                else:
                    expected = (SCALE.index(clef_signs[-1])-3) % 7
                    clef_signs.append(SCALE[expected])  # should we check correctness? or assume notes are correct
                are_sharps = False
            else:
                local_flats.append(find_tone(clef, line_number)[0])
        elif sign[2] == "natural":
            if clef is None:
                print("ERROR: No clef")
                return None
            x, y = sign_center(sign)
            line_number = what_line(x, y, lines_data, mean_distance_between_lines)
            print("natural on ", line_number)
            tone = find_tone(clef, line_number)[0]
            if tone in clef_signs:
                locally_deleted_signs.append(tone)
            if tone in local_sharps:
                local_sharps.remove(tone)
            if tone in local_flats:
                local_flats.remove(tone)
        elif sign[2].startswith("note"):
            if clef is None:
                print("ERROR: No clef")
                return None
            x, y = sign_center(sign)
            y -= 2
            line_number = what_line(x, y, lines_data, mean_distance_between_lines)
            print("note on ", line_number)
            tone, octave = find_tone(clef, line_number)
            if tone in local_sharps or (tone in clef_signs and are_sharps and tone not in locally_deleted_signs):
                tone += '#'
            if tone in local_flats or (tone in clef_signs and not are_sharps and tone not in locally_deleted_signs):
                tone += 'b'
            prefix, note_type = sign[2].split('_')
            note_info = (int(note_type), tone, octave)
            sounds.append(note_info)
            if is_repetition:
                repetition.append(note_info)
        elif sign[2].startswith("rest"):
            prefix, rest_type = sign[2].split('_')
            rest = (int(rest_type), "r")
            sounds.append(rest)
        elif sign[2] == "line":
            locally_deleted_signs.clear()
            local_sharps.clear()
            local_flats.clear()
        elif sign[2] == "repetition_start":
            is_repetition = True
            locally_deleted_signs.clear()
            local_sharps.clear()
            local_flats.clear()
        elif sign[2] == "repetition_end":
            if not is_repetition:
                repetition = sounds.copy()
            is_repetition = False
            for sound in repetition:
                sounds.append(sound.copy())
            repetition.clear()
            locally_deleted_signs.clear()
            local_sharps.clear()
            local_flats.clear()
        elif sign[2] == "fermata":
            time, tone, octave = note.split('|')
            time_number = int(time)
            time_number /= 2
            note = str(time_number) + '|' + tone + '|' + octave
        else:
            print("Unknown sign")
        previous_sign = sign
    print("Clef: ", clef)
    print("Clef ", "sharps:" if are_sharps else "flats: ", clef_signs)

    return sounds


def check_is_natural(coords, img):
    x1 = coords[0][0]
    x2 = coords[1][0]
    y = coords[0][1]
    is_black = False
    count_black_lines = 0
    for x in range(x1, x2):
        if img[y][x] == 0:
            if not is_black:
                is_black = True
                count_black_lines += 1
        else:
            is_black = False
    if count_black_lines == 2:
        return coords[0], coords[1], "sharp"
    elif count_black_lines == 1:
        return coords[0], coords[1], "natural"
