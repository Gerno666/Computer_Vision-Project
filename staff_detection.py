import cv2
import statistics
import sys  # just for the progress printing in correcting non perfectly horizontal lines
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from general_functions import show


# Returns array of pixels of one line (not perfectly horizontal)
def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points



# detects some lines (usually just parts of real lines) also the vertical ones
def basic_detection(binary):
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    # Hough's transformation
    detected = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    return detected


# merges parts (detected) of the same (real) line
def merge_line_parts(line_parts):
    x1s = []
    x2s = []
    y1s = []
    y2s = []
    for line in line_parts:
        x1s.append(line[0])
        x2s.append(line[2])
        y1s.append(line[1])
        y2s.append(line[3])
    start_y = int(np.mean(y1s))
    end_y = int(np.mean(y2s))
    start_x = np.min(x1s)
    end_x = np.max(x2s)
    return [start_x, start_y, end_x, end_y]


# it calculates factors a and b of the line equation: y = ax + b
# segment is a part of a line, it is stored as array containing coordinates [x1, y1, x2, y2]
def find_line_equation(segment):
    points = [(segment[0], segment[1]), (segment[2], segment[3])]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    a, b = lstsq(A, y_coords, rcond=None)[0]
    return a, b


# for all detected parts of lines it 'groups' them by the 'belonging' to actual lines and merge parts of the same line
def find_actual_lines(all_lines, line_surrounding_threshold):
    all_lines = sorted(all_lines, key=lambda x: x[0][1], reverse=False)  # sort lines from highest to lowest
    actual_lines = []
    last_position = all_lines[0][0][1]
    same_line = [all_lines[0][0]]
    for i in range(1, len(all_lines)):
        if last_position - line_surrounding_threshold < all_lines[i][0][1] < last_position + line_surrounding_threshold:
            same_line.append(all_lines[i][0])
        else:
            one_line = merge_line_parts(same_line)
            actual_lines.append(one_line)
            same_line.clear()
            same_line.append(all_lines[i][0])
        last_position = all_lines[i][0][1]

    one_line = merge_line_parts(same_line)
    actual_lines.append(one_line)
    return actual_lines


# corrects lines to cover real lines (black pixels)
def correct_lines_positions(lines, binary_image):
    for line in lines:
        is_horizontal = line[1] == line[3]
        black = 0
        j = line[1]
        if is_horizontal:
            for i in range(line[0], line[2]):
                if binary_image[j][i] == 0:
                    black += 1
        else:
            points = get_line(line[0], line[1], line[2], line[3])
            for point in points:
                if point[1] < binary_image.shape[0] and point[0] < binary_image.shape[1] and binary_image[point[1]][point[0]] == 0:
                    black += 1
        if black < (line[2] - line[0]) * 0.8:
            if is_horizontal:
                blacks = []
                for i in range(line[0], line[2]):
                    for j in range(0, 10):
                        if binary_image[line[1] + j][i] == 0:
                            blacks.append(j)
                            break
                        if binary_image[line[1] - j][i] == 0:
                            blacks.append(-j)
                            break
                how_far_to_line = 0
                for i in range(-10, 10):
                    if i == 0:
                        continue
                    if blacks.count(i) > how_far_to_line:
                        how_far_to_line = i
                line[1] += how_far_to_line
                line[3] += how_far_to_line
            else:
                blacks1 = []
                blacks2 = []
                points = get_line(line[0], line[1], line[2], line[3])
                for point in points[:int(len(points) / 4)]:
                    i = point[0]
                    j = point[1]
                    for k in range(0, 10):
                        if j+k < binary_image.shape[0] and i < binary_image.shape[1] and binary_image[j + k][i] == 0:
                            blacks1.append(k)
                            break
                        if j-k < binary_image.shape[0] and i < binary_image.shape[1] and binary_image[j - k][i] == 0:
                            blacks1.append(-k)
                            break
                for point in points[:len(points) - int(len(points) / 4)]:
                    i = point[0]
                    j = point[1]
                    for k in range(0, 10):
                        if j+k < binary_image.shape[0] and i < binary_image.shape[1] and binary_image[j + k][i] == 0:
                            blacks2.append(k)
                            break
                        if j-k < binary_image.shape[0] and i < binary_image.shape[1] and binary_image[j - k][i] == 0:
                            blacks2.append(-k)
                            break
                how_far_to_line = 0
                for i in range(-10, 10):
                    if i == 0:
                        continue
                    if blacks1.count(i) > how_far_to_line:
                        how_far_to_line = i
                line[1] += how_far_to_line
                how_far_to_line = 0
                for i in range(-10, 10):
                    if i == 0:
                        continue
                    if blacks2.count(i) > how_far_to_line:
                        how_far_to_line = i
                line[3] += how_far_to_line
    return lines


# extends a line to cover all image width's (keeping the same angle)
def full_line(segment, last_x):
    a, b = find_line_equation(segment)
    y1 = b
    y2 = a*last_x + b
    return 0, y1, last_x, y2


# extends detected lines to cover all length of real lines in the picture
def extend_lines(lines, binary_image, perfectly_horizontal):
    if perfectly_horizontal:
        for line in lines:
            for i in range(line[0], 1, -1):
                if binary_image[line[1]][i - 1] == 0:
                    line[0] = i - 1
                else:
                    break
            for i in range(line[2], len(binary_image[0]) - 2):
                if binary_image[line[3]][i + 1] == 0:
                    line[2] = i + 1
                else:
                    break
    else:
        for line in lines:
            x1, y1, x2, y2 = full_line(line, binary_image.shape[1] - 1)
            line[0] = x1
            line[1] = y1
            line[2] = x2
            line[3] = y2

    return lines


def longest_monotonic_subsequence(sequence, increasing):
    n = len(sequence)
    dp = [1] * n

    for i in range(1, n):
        for j in range(0, i):
            if increasing:
                if sequence[i] >= sequence[j] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
            else:
                if sequence[i] <= sequence[j] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1

    max_length = max(dp)
    last_index = dp.index(max_length)
    longest_subsequence = [sequence[last_index]]

    for i in range(last_index - 1, -1, -1):
        if increasing:
            if sequence[i] <= longest_subsequence[-1] and dp[i] == max_length - 1:
                longest_subsequence.insert(0, sequence[i])
                max_length -= 1
        else:
            if sequence[i] >= longest_subsequence[-1] and dp[i] == max_length - 1:
                longest_subsequence.insert(0, sequence[i])
                max_length -= 1

    return longest_subsequence


# after extending not perfectly horizontal lines it corrects their angle to fit actual black line
def correct_extended_lines(lines, binary_image, threshold, distance_between_lines):
    progress = 0
    for line in lines:
        sys.stdout.write('\r' + str(int(progress/len(lines)*100)) + "% done")
        points = get_line(line[0], line[1], line[2], line[3])
        how_far_black = []
        for point in points:
            i = 0
            while i < distance_between_lines:
                if 0 < point[1]+i < binary_image.shape[0] and binary_image[point[1]+i][point[0]] == 0:
                    how_far_black.append(i)
                    break
                if 0 < point[1]-i < binary_image.shape[0] and binary_image[point[1] - i][point[0]] == 0:
                    how_far_black.append(-i)
                    break
                i += 1

        increasing = longest_monotonic_subsequence(how_far_black, True)
        descending = longest_monotonic_subsequence(how_far_black, False)

        if len(increasing) > len(descending):
            accepted_values = []
            for i in range(1, len(increasing)):
                if increasing[i]-increasing[i-1] <= threshold:
                    accepted_values.append(increasing[i])
            if len(accepted_values) != 0:
                lowest = min(accepted_values)
                highest = max(accepted_values)
                line[1] += lowest
                line[3] += highest
        else:
            accepted_values = []
            for i in range(1, len(descending)):
                if descending[i - 1] - descending[i] <= threshold:
                    accepted_values.append(descending[i])
            if len(accepted_values) != 0:
                highest = max(accepted_values)
                lowest = min(accepted_values)
                line[1] += highest
                line[3] += lowest
        progress += 1
    sys.stdout.write('\r' + "100% done\n")
    return lines


def merging_not_horizontal(lines, threshold):
    lines_with_params = []
    for line in lines:
        a, b = find_line_equation([line[0], line[1], line[2], line[3]])
        lines_with_params.append([line, a, b])

    lines_with_params = sorted(lines_with_params, key=lambda x: x[2], reverse=False)

    actual_lines = []
    lines_to_merge = []
    previous_line = lines_with_params[0]
    lines_to_merge.append(previous_line)
    for line in lines_with_params[1:]:
        if abs(line[2]-previous_line[2]) < threshold:
            lines_to_merge.append(line)
        else:
            if len(lines_to_merge) != 1:
                x1s = []
                x2s = []
                y1s = []
                y2s = []
                for ln in lines_to_merge:
                    x1s.append(ln[0][0])
                    x2s.append(ln[0][2])
                    y1s.append(ln[0][1])
                    y2s.append(ln[0][3])
                x1 = min(x1s)
                y1 = int(np.mean(y1s))
                x2 = max(x2s)
                y2 = int(np.mean(y2s))
                actual_lines.append([x1, y1, x2, y2])
            else:
                actual_lines.append(previous_line[0])

            lines_to_merge.clear()
            previous_line = line
            lines_to_merge.append(previous_line)

    if len(lines_to_merge) != 1:
        x1s = []
        x2s = []
        y1s = []
        y2s = []
        for ln in lines_to_merge:
            x1s.append(ln[0][0])
            x2s.append(ln[0][2])
            y1s.append(ln[0][1])
            y2s.append(ln[0][3])
        x1 = min(x1s)
        y1 = int(np.mean(y1s))
        x2 = max(x2s)
        y2 = int(np.mean(y2s))
        actual_lines.append([x1, y1, x2, y2])
    else:
        actual_lines.append(previous_line[0])

    return actual_lines


# detects all the horizontal lines ang organizes them as staffs
def staff_detection(bin_img, dist_between_lines):
    # detect some lines with Hough's transform - some segments detected
    detected_lines = basic_detection(bin_img)

    # reject detected vertical lines
    accepted_lines = []  # only horizontal lines
    if detected_lines is not None:
        threshold = 10  # angle threshold
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            # calculate line's angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # check
            if abs(angle) < threshold or abs(angle - 180) < threshold:
                accepted_lines.append(line)
    ###
    # show accepted lines
    img_to_show = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    for line in accepted_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_to_show, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("accepted", img_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ###

    detected_lines = None  # remove useless array (free memory)

    # check if all the lines are perfectly horizontal or not
    perfectly_horizontal = True
    for line in accepted_lines:
        if line[0][1] != line[0][3]:
            perfectly_horizontal = False
            break

    actual_lines = None  # result array

    if perfectly_horizontal:
        # calculate threshold - pixels above and under the line
        threshold = dist_between_lines/2
        if threshold - int(threshold) == 0.5:
            threshold = np.ceil(threshold)
        else:
            threshold = round(threshold)

        # merge lines' parts into whole lines
        merged_lines = find_actual_lines(accepted_lines, threshold)
        # show(merged_lines, "merged", bin_img)
        accepted_lines = None  # remove useless array (free memory)

        # correct detected lines to cover real lines (black pixels)
        corrected_lines = correct_lines_positions(merged_lines, bin_img)
        # show(corrected_lines, "corrected", bin_img)
        merged_lines = None  # remove useless array (free memory)

        # extending lines (to cover all the real black lines)
        actual_lines = extend_lines(corrected_lines, bin_img, True)
        show(actual_lines, "lines found", bin_img)
        corrected_lines = None  # remove useless array (free memory)

    else:
        lines_segments = []
        for line in accepted_lines:
            lines_segments.append(line[0])

        # correct detected lines to cover real lines (black pixels)
        corrected_segments = correct_lines_positions(lines_segments, bin_img)
        # show(corrected_segments, "corrected", bin_img)
        lines_segments = None  # remove useless array (free memory)

        # extending lines (to cover all width of the image)
        full_lines = extend_lines(corrected_segments, bin_img, False)
        show(full_lines, "extended", bin_img)
        corrected_segments = None  # remove useless array (free memory)

        print("Correcting positions... (it may take a while)")
        # correct again full length lines to cover actual black lines as well as possible
        approximated_lines = correct_extended_lines(full_lines, bin_img, 4, dist_between_lines)
        show(approximated_lines, "corrected again", bin_img)
        full_lines = None  # remove useless array (free memory)

        # merge lines that are duplicated
        merged_lines = merging_not_horizontal(approximated_lines, dist_between_lines/3)
        show(merged_lines, "merged", bin_img)
        approximated_lines = None  # remove useless array (free memory)

        # correct position of merged lines to cover actual black lines
        actual_lines = correct_lines_positions(merged_lines, bin_img)
        # show(actual_lines, "lines found", bin_img)
        merged_lines = None  # remove useless array (free memory)

    print(len(actual_lines), "lines found")

    # group lines by 5
    staffs = []
    staff = []
    for line in actual_lines:
        staff.append(line)
        if len(staff) == 5:  # check the distance between lines - if it's similar, then the staff is correct
            previous_distance = None
            is_correct = True
            for i in range(1, 4):
                dist1 = staff[i][1] - staff[i-1][1]
                dist2 = staff[i][3] - staff[i-1][3]
                if dist1 == 0 or abs(dist1-dist_between_lines) > dist_between_lines/2 or dist2 == 0 or abs(dist2-dist_between_lines) > dist_between_lines/2 or abs(dist1-dist2) > dist_between_lines/2 or dist1 < dist_between_lines*0.75 or dist2 < dist_between_lines*0.75:
                    staff.remove(staff[0])
                    is_correct = False
                    break
                if previous_distance is not None:
                    if abs(dist1 - previous_distance) > dist_between_lines/4:
                        staff.remove(staff[0])
                        is_correct = False
                        break
                previous_distance = dist1
            if is_correct:
                staffs.append(staff.copy())
                staff.clear()
    return staffs


def estimate_line_distance(bin_img):
    estimated_distance = []
    for j in range(bin_img.shape[1]):
        white_counter = 0
        previous_white_field = None
        for i in range(bin_img.shape[0]):
            if bin_img[i][j] == 255:
                white_counter += 1
            else:
                if white_counter != 0:
                    if previous_white_field is not None:
                        if abs(white_counter - previous_white_field) < 2:
                            estimated_distance.append(white_counter)
                    previous_white_field = white_counter
                white_counter = 0
    distance = statistics.mode(estimated_distance)
    print("Estimated distance between lines: ", distance)
    return distance
