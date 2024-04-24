import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import random
import math
from enum import Enum

# A4 size in pixels at 300dpi
A4_WIDTH = 2480
A4_HEIGHT = 3508


def recursive_mask(masked: list[bool], locked: list[bool], max_masked: int):
    if len([m for m in masked if m]) >= max_masked:
        return masked

    # identify locations that can be masked
    n = len(masked)
    maskable = [False] * n
    left_pass = [False] * n
    right_pass = [False] * n

    for i in range(len(masked)):
        if not locked[i] and not masked[i]:
            if i % 2 == 0:
                # a circle
                if i > 1 and not masked[i - 2] and not masked[i - 1]:
                    left_pass[i] = True
                if i < len(masked) - 2 and not masked[i + 1] and not masked[i + 2]:
                    right_pass[i] = True
                if left_pass[i] or right_pass[i]:
                    maskable[i] = True
            else:
                # a segment
                maskable[i] = True

    choices = [i for i in range(n) if maskable[i]]
    # print(f"maskable choices: {choices}")
    if not choices:
        return masked

    selected = random.choice(choices) if len(choices) > 1 else choices[0]
    # print(f"selected choice is {selected}")

    if selected % 2 == 0:
        # a circle again
        choices = []
        if left_pass[selected]:
            choices.append(-1)
        if right_pass[selected]:
            choices.append(1)

        assert choices

        lock_choice = random.choice(choices) if len(choices) > 1 else choices[0]

        locked[selected + lock_choice] = True
        locked[selected + lock_choice * 2] = True

    # nothing special needs to be done for segments
    masked[selected] = True
    return recursive_mask(masked, locked, max_masked)


class Op(Enum):
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"


class Transform:

    def __init__(self, op: Op, e: int, masked: bool = False):
        self.op = op
        self.e = e
        self.masked = masked

    def apply_to(self, v: int) -> int:
        e = self.e
        match self.op:
            case Op.ADD:
                return v + e
            case Op.SUB:
                return v - e
            case Op.MUL:
                return v * e
            case Op.DIV:
                return v // e

    def __repr__(self):
        e = self.e if not self.masked else "  "
        match self.op:
            case Op.ADD:
                return f"+{e}"
            case Op.SUB:
                return f"-{e}"
            case Op.MUL:
                return f"×{e}"
            case Op.DIV:
                return f"÷{e}"


def compute_valid_xforms(v: int) -> list[Transform]:
    adds = [Transform(Op.ADD, e) for e in range(10) if abs(v + e) < 20]
    subs = [Transform(Op.SUB, e) for e in range(10) if abs(v - e) < 20]
    muls = [Transform(Op.MUL, e) for e in range(1, 5) if abs(v * e) < 20]
    divs = [Transform(Op.DIV, e) for e in range(1, 6) if v % e == 0 and v > 0]
    return adds + subs + muls + divs


class Snake:

    def __init__(self):
        self.values = []
        self.segment_angles = []
        self.segment_lengths = []
        self.operations = []

    def generate_snake(
        self, length: int, angle_range: (int, int), segment_range: (int, int)
    ):
        angles = [random.randint(angle_range[0], angle_range[1]) for _ in range(length)]
        start_angle_dir = random.choice([-1, 1])
        self.segment_angles = [
            a * start_angle_dir * ((i % 2) * 2 - 1) for i, a in enumerate(angles)
        ]
        self.segment_lengths = [
            random.randint(segment_range[0], segment_range[1]) for _ in range(length)
        ]

        start_value = random.randint(-5, 10)
        values = [start_value] + [0] * (length * 2)
        for i in range(1, length * 2 + 1, 2):
            valid_operations = compute_valid_xforms(values[i - 1])
            operation = random.choice(valid_operations)
            values[i] = operation  # Assign operation
            values[i + 1] = operation.apply_to(values[i - 1])
        self.values = values

    def mask_values(self, max_masked: int):
        n = len(self.values)
        masked0, locked0 = [False] * n, [False] * n
        mask = recursive_mask(masked0, locked0, max_masked)

        mask_xform = lambda t: Transform(t.op, t.e, True)
        self.masked = [
            v if not mask[i] else (mask_xform(v) if i % 2 == 1 else None)
            for i, v in enumerate(self.values)
        ]

    def compute_bounding_box_and_origin(self):
        padding = 80
        left, right, top, bottom = 0, 0, 0, 0

        # assume origin always at 0,0 and snakes always go right
        x, y = 0, 0
        origin_x, origin_y = padding // 2, 0
        for angle, length in zip(self.segment_angles, self.segment_lengths):
            angle_rad = angle * math.pi / 180
            x += length * math.cos(angle_rad)
            y += length * math.sin(angle_rad)
            right = max(right, x)
            if y > top:
                top = y

            if y < bottom:
                origin_y += bottom - y
                bottom = y

        self.bbox = math.ceil(right - left + padding), math.ceil(top - bottom + padding)
        self.origin = (origin_x, round(origin_y))

    def paint_circle(self, ax, x, y, radius, value):
        # paint circle
        circle = plt.Circle((x, y), radius, color="blue", fill=False)
        ax.add_artist(circle)
        if value is not None:
            ax.text(
                x,
                y,
                str(value),
                color="black",
                fontsize=10,
                ha="center",
                va="center",
            )

    def paint_line(self, ax, x0, y0, x1, y1, radius, op):
        ax.add_artist(plt.Line2D((x0, x1), (y0, y1)))

        # text base position
        base_x, base_y = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        px, py = -dy, dx
        nxy = math.sqrt(px**2 + py**2)

        # move in perpendicular direction only distance
        dp = radius * 1.2
        renorm = dp / nxy
        px, py = px * renorm, py * renorm

        cx, cy = base_x + px, base_y + py

        circle = plt.Circle((cx, cy), radius * 0.9, color="green", fill=False)
        ax.add_artist(circle)
        if op is not None:
            ax.text(
                cx,
                cy,
                str(op),
                color="black",
                fontsize=10,
                ha="center",
                va="center",
            )

    def paint(self, ax, bbox_x, bbox_y, masked: bool):
        values = self.masked if masked else self.values
        circle_radius = 60  # radius of each circle
        x = bbox_x + self.origin[0] + circle_radius
        y = bbox_y + self.origin[1] + circle_radius

        for i, (length, angle) in enumerate(
            zip(self.segment_lengths, self.segment_angles)
        ):
            circle_value, segment_op = values[2 * i], values[2 * i + 1]
            angle_rad = angle * math.pi / 180

            self.paint_circle(ax, x, y, circle_radius, circle_value)

            # compute start/end position of circle connector
            x0 = x + circle_radius * math.cos(angle_rad)
            y0 = y + circle_radius * math.sin(angle_rad)

            line_length = length - circle_radius
            x1 = x + line_length * math.cos(angle_rad)
            y1 = y + line_length * math.sin(angle_rad)

            self.paint_line(ax, x0, y0, x1, y1, circle_radius, segment_op)

            # compute position of next circle
            next_x = x + length * math.cos(angle_rad)
            next_y = y + length * math.sin(angle_rad)

            x, y = next_x, next_y

        # add last circle
        circle_value = values[-1]
        self.paint_circle(ax, x, y, circle_radius, circle_value)

    def __str__(self):
        length = len(self.segment_lengths)
        return f"Snake({length}, {self.values}, {self.masked}, {self.segment_lengths}, {self.segment_angles}, {self.bbox}, {self.origin})"


def create_snake(length: int):
    snake = Snake()
    snake.generate_snake(length, (0, 30), (250, 400))
    snake.compute_bounding_box_and_origin()
    # don't limit number of masks
    snake.mask_values(length)
    return snake


def create_image(snakes: list[Snake], title: str, masked: bool, pdf_pages: PdfPages):
    fig, ax = plt.subplots(figsize=(A4_WIDTH / 300, A4_HEIGHT / 300), dpi=300)
    ax.set_xlim(0, A4_WIDTH)
    ax.set_ylim(0, A4_HEIGHT)
    ax.axis("off")

    ax.text(
        A4_WIDTH / 2,
        A4_HEIGHT + 50,
        title,
        color="black",
        fontsize=20,
        ha="center",
        va="center",
    )

    # Randomly place snakes
    i = 0
    x, y = 50, 50
    pw, ph = A4_WIDTH - 50, A4_HEIGHT - 100
    row_height = 0
    pad = 100
    while y < ph:
        print(
            f"Placing row at y={y} with x={x}, paper dims w x h {pw, ph} and row_height={row_height}"
        )
        row_height = 0
        while x < pw:
            snake = snakes[i]
            bbox = snake.bbox
            print(f"Snake bbox is {snake.bbox}")

            # can place snake on row?
            if x + bbox[0] < pw and y + bbox[1] < ph:
                print(f"Placing snake {snake}")
                snake.paint(ax, x, y, masked)
                row_height = max(row_height, bbox[1])
                x += bbox[0] + pad
                i += 1
            else:
                break

        y = y + row_height + pad
        x = 50

    plt.gca().set_aspect("equal", adjustable="box")
    # plt.savefig(path)
    pdf_pages.savefig(fig)


# Example usage
snakes = [create_snake(4) for _ in range(100)]

pdf_pages = PdfPages("snakes.pdf")
title = "Hadi pro Adélku"
create_image(snakes, title, True, pdf_pages)
create_image(snakes, title + ": řešení", False, pdf_pages)
pdf_pages.close()
