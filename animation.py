from manim import *
import numpy as np
from math import atan2, cos, sin, pi
from func import func, Point
from func import display_data
from func import distance, safety_distance, findTimeForDistance, t_for_stop


# стандартные библиотеки для кэша
import pickle
import hashlib
import os

CACHE_FILE = "func_cache.pkl"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            # если файл повреждён — начинать с пустого кэша
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def serialize_point(p):
    """
    Пытаемся получить числовые координаты точки в предсказуемом формате.
    Поддерживаем: объект с атрибутами x,y; индексируемый (tuple/list); fallback -> repr.
    """
    # объект с .x и .y
    try:
        x = float(getattr(p, "x"))
        y = float(getattr(p, "y"))
        return (x, y)
    except Exception:
        pass

    # индексируемый (tuple/list)
    try:
        if (hasattr(p, "__len__") and len(p) >= 2):
            x = float(p[0])
            y = float(p[1])
            return (x, y)
    except Exception:
        pass

    # fallback
    return (repr(p),)

def make_cache_key(points, *args, **kwargs):
    """
    Создаёт детерминированное представление входных данных.
    points: ожидаем словарь, сортируем по ключам.
    args/kwargs: включаем как примитивы (repr для всего остального).
    Возвращаем sha256 hex digest строки представления.
    """
    # сериализуем points в список пар (k, x, y) отсортированных по ключу
    pts_list = []
    try:
        for k in sorted(points.keys()):
            p = points[k]
            sp = serialize_point(p)
            pts_list.append((k, sp))
    except Exception:
        # в неожиданных случаях — просто взять repr
        pts_list = repr(points)

    # сериализация args
    args_list = []
    for a in args:
        try:
            # примитивы: числа/строки/логические
            if isinstance(a, (int, float, str, bool, type(None))):
                args_list.append(a)
            else:
                args_list.append(repr(a))
        except Exception:
            args_list.append(repr(a))

    # kwargs — отсортировать по ключу
    kwargs_list = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        try:
            if isinstance(v, (int, float, str, bool, type(None))):
                kwargs_list.append((k, v))
            else:
                kwargs_list.append((k, repr(v)))
        except Exception:
            kwargs_list.append((k, repr(v)))

    key_struct = ("points", tuple(pts_list), "args", tuple(args_list), "kwargs", tuple(kwargs_list))
    key_bytes = repr(key_struct).encode("utf-8")
    return hashlib.sha256(key_bytes).hexdigest()

# Загружаем кэш один раз при старте модуля
_CACHE = load_cache()

def cached_func(points, *args, **kwargs):
    """
    Обёртка вокруг func: сначала смотрим в кэш по ключу,
    если нет — вызываем func, сохраняем результат и возвращаем.
    """
    global _CACHE
    key = make_cache_key(points, *args, **kwargs)

    if key in _CACHE:
        return _CACHE[key]

    # если нет — вычисляем, сохраняем и возвращаем
    result = func(points, *args, **kwargs)
    _CACHE[key] = result
    try:
        save_cache(_CACHE)
    except Exception:
        # не фатально, игнорируем ошибки сохранения
        pass
    return result




class OneCar(MovingCameraScene):
    def construct(self):

        self.camera.frame.scale(6.0)

        # ---------- сетка ----------
        line_spacing = 3.0
        line_length = 11

        data_left_turn = {
            5: [
            [0,0,0,3],
            [0,0,0,0],
            [1,0,0,0],
            [0,0,1,0],
            [0,0,0,0],
            [1,0,0,0]
            ],

            10: [
            [0,2,0,0],
            [1,0,0,2],
            [0,1,0,0],
            [1,0,1,1],
            [0,0,1,0],
            [1,0,0,1]
            ],

            15: [
            [2,0,1,0],
            [1,1,0,1],
            [0,2,0,1],
            [1,0,2,0],
            [1,0,0,2],
            [0,1,1,1]
            ],

            20: [
            [2,1,1,0],
            [0,3,0,1],
            [1,2,0,1],
            [0,2,1,1],
            [2,0,2,0],
            [1,1,2,0]
            ],

            25: [
            [3,0,1,2],
            [0,2,1,2],
            [1,3,0,1],
            [2,0,2,1],
            [1,1,3,0],
            [2,1,0,2]
            ],

            30: [
            [2,2,0,3],
            [1,3,1,1],
            [2,0,2,2],
            [3,1,1,1],
            [0,2,3,0],
            [1,2,1,3]
            ],

            35: [
            [3,1,2,1],
            [0,3,2,1],
            [2,2,0,3],
            [1,3,1,2],
            [2,1,3,0],
            [1,2,2,2]
            ],

            40: [
            [0,3,1,2],
            [2,0,3,1],
            [1,2,0,2],
            [3,1,2,0],
            [2,1,1,3],
            [1,2,0,1]
            ],

            45: [
            [3,2,1,2],
            [1,3,2,1],
            [2,2,3,0],
            [3,1,2,2],
            [0,3,2,2],
            [2,1,3,1]
            ],

            50: [
            [4,1,2,3],
            [2,3,1,3],
            [3,0,4,1],
            [1,4,2,2],
            [3,2,3,0],
            [2,3,1,3]
            ],

            55: [
            [3,3,2,3],
            [4,1,3,2],
            [2,4,1,3],
            [3,2,4,1],
            [1,3,3,3],
            [4,2,2,2]
            ],

            60: [
            [3,4,2,3],
            [1,4,3,2],
            [4,1,4,2],
            [2,3,4,1],
            [3,2,3,4],
            [4,1,2,4]
            ],

            65: [
            [4,3,4,2],
            [2,4,3,3],
            [3,1,4,4],
            [4,2,3,4],
            [1,4,4,3],
            [3,4,2,4]
            ],

            70: [
            [4,3,3,4],
            [3,4,2,3],
            [4,3,4,3],
            [2,4,3,4],
            [3,4,4,2],
            [4,3,2,4]
            ],

            75: [
            [4,3,4,4],
            [3,4,3,4],
            [4,2,4,4],
            [3,4,4,3],
            [4,4,2,4],
            [3,4,3,4]
            ],

            80: [
            [4,1,3,4],
            [0,4,2,3],
            [4,3,0,4],
            [2,4,1,3],
            [4,0,3,4],
            [2,3,4,0]
            ]
        }

        left_turn = data_left_turn[50]

        for i in range(7):
            x = line_spacing * (i - 3)
            self.add(
                Line(UP * line_length, DOWN * line_length)
                .shift(RIGHT * x)
                .set_color(WHITE)
            )

        for i in range(7):
            y = line_spacing * (i - 3)
            self.add(
                Line(RIGHT * line_length, LEFT * line_length)
                .shift(DOWN * y)
                .set_color(WHITE)
            )

        # ---------- функция положения ----------
        def build_position_function(data):
            segments = []
            for key in sorted(data.keys()):
                for seg in data[key]:
                    if seg.t1 > seg.t0:
                        segments.append(seg)

            segments.sort(key=lambda s: s.t0)

            def s_of_t(t):
                for seg in segments:
                    if seg.t0 <= t < seg.t1:
                        if seg.nameAccel == "C":
                            return seg.s0 + seg.v0 * (t - seg.t0)
                        else:
                            k = 1.0 if seg.nameAccel == "I" else -1.0
                            return distance(
                                t,
                                seg.a0,
                                seg.v0,
                                seg.s0,
                                seg.t0,
                                k
                            )
                return segments[-1].s1

            return s_of_t, segments
        

        def build_follow_function(s_of_t, segments):
            """
            f(t) = s(t) + safety_distance(v(t), a0, v0, t0, k)
            Здесь safety_distance ожидает первым аргументом скорость v, затем параметры сегмента.
            (в твоём окружении сигнатура может быть другая — оставил вызов в том стиле,
            который был в твоём последнем варианте кода)
            """
            def f_of_t(t):
                if not segments:
                    return 0.0

                for seg in segments:
                    if seg.t0 <= t < seg.t1:
                        # получаем локальные значения
                        s = s_of_t(t)
                        k = 1.0 if getattr(seg, "nameAccel", "") == "I" else (-1.0 if getattr(seg, "nameAccel", "") == "D" else 0.0)

                        # вызываем safety_distance с параметрами, которые используются в коде
                        safe = safety_distance(t, seg.a0, seg.v0, seg.t0, k)
                        return s

                # если t >= конца -- возьмём последний сегмент
                last = segments[-1]
                s_last = s_of_t(last.t1)
                k_last = 1.0 if getattr(last, "nameAccel", "") == "I" else (-1.0 if getattr(last, "nameAccel", "") == "D" else 0.0)
                safe_last = safety_distance(t, last.a0, last.v0, last.t0, k_last)
                return s_last
            
            def t_of_stop(t):
                if not segments:
                    return 0.0

                for seg in segments:
                    if seg.t0 <= t < seg.t1:
                        # получаем локальные значения
                        k = 1.0 if getattr(seg, "nameAccel", "") == "I" else (-1.0 if getattr(seg, "nameAccel", "") == "D" else 0.0)

                        # вызываем safety_distance с параметрами, которые используются в коде
                        t = t_for_stop(t, seg.a0, seg.v0, seg.t0, k)
                        return t

                # если t >= конца -- возьмём последний сегмент
                last = segments[-1]
                k_last = 1.0 if getattr(last, "nameAccel", "") == "I" else (-1.0 if getattr(last, "nameAccel", "") == "D" else 0.0)
                t_last = t_for_stop(t, last.a0, last.v0, last.t0, k_last)
                return t_last
            
            return f_of_t, t_of_stop

        def invert_monotonic_function(f_of_t, s_target, t_min, t_max, tol=1e-6, max_iter=100):
            """
            Находит t такое, что f_of_t(t) ≈ s_target.
            
            Предполагается, что f_of_t монотонно возрастает на [t_min, t_max].
            """
            f_min = f_of_t(t_min)
            f_max = f_of_t(t_max)

            # Проверка диапазона
            if not (f_min <= s_target <= f_max):
                # Если целевое значение вне диапазона — выбросим подробную ошибку
                raise ValueError(f"s_target ({s_target}) вне диапазона [{f_min}, {f_max}] функции на данном интервале")

            left = t_min
            right = t_max

            for _ in range(max_iter):
                mid = 0.5 * (left + right)
                f_mid = f_of_t(mid)

                if abs(f_mid - s_target) < tol:
                    return mid

                if f_mid < s_target:
                    left = mid
                else:
                    right = mid

            # если не достигли точности — возвращаем лучшее приближение
            return 0.5 * (left + right)

        # ---------- создаём машины ----------
        tracker = ValueTracker(0)
        total_time = 0

        # ---------- таймер (показывает десятые секунды) ----------
        # отображает "t = X.X s" в правом верхнем углу
        time_number = DecimalNumber(
            0,
            num_decimal_places=1,
            include_sign=False,
            unit=" s"
        ).scale(3.0)

        time_label = Text("t =", font_size=84)
        time_display = VGroup(time_label, time_number)
        time_display.arrange(RIGHT, buff=0.15)

        # фиксируем позицию ОДИН раз
        time_display.move_to(24 * LEFT + 15 * UP)
        time_display.set_z_index(1000)

        # обновляем только значение числа
        time_number.add_updater(lambda m: m.set_value(tracker.get_value()))

        self.add(time_display)

        conflict_points = {}

        for index in range(1):
            print(index)
            for direction in ("left", "up", "right", "down"):
                if direction == "left":
                    if index == 0:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][0] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 1.5
                                        ).shift(UP * 1.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 1.5
                                        ).shift(UP * 4.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 4.5
                                        ).shift(UP * 4.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 1.5
                                        ).shift(UP * 7.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 0.0
                                    self.add(car)

                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой (начало дуги)
                                    p2 = np.array(line2.get_start())   # начало второй прямой (конец дуги)

                                    # центр четверти окружности (подходящий для перпендикулярных отрезков)
                                    center = np.array([p1[0], p2[1], 0.0])
                                    # радиус:
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle = atan2(p2[1] - center[1], p2[0] - center[0])


                                    # нормализуем span в диапазон [-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                    # --- используем cached_func / s_func как раньше ---
                                    points = {
                                        1: Point(0.1, 3.0),
                                        2: Point(1.0, 6.0),
                                        3: Point(1.0, 9.0),
                                        4: Point(1.0, 12.0),
                                        5: Point(1.0, 15.0),
                                        6: Point(1.0, 18.0)
                                    }
                                    data = cached_func(points, total_path_length)
                                    s_func, segments = build_position_function(data)

                                    if left_turn[index][0] == 1:
                                        if i == j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i] = conflict_points[index - 1]["down"].get(i, dict())
                                            
                                            conflict_points[index]["down"][i][12] = t12
                                            conflict_points[index]["down"][i][15] = t15
                                            conflict_points[index]["down"][i][18] = t18
                                    elif left_turn[index][0] == 2:
                                        if i == j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i] = conflict_points[index - 1]["down"].get(i, dict())
                                            
                                            conflict_points[index]["down"][i][12] = t12
                                            conflict_points[index]["down"][i][15] = t15
                                            conflict_points[index]["down"][i][18] = t18
                                        elif i == 1 and j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i] = conflict_points[index - 1]["down"].get(i, dict())
                                            
                                            conflict_points[index]["down"][i][12] = t12
                                            conflict_points[index]["down"][i][15] = t15
                                            conflict_points[index]["down"][i][18] = t18
                                    elif left_turn[index][0] == 3:
                                        if i == j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9
                                        elif i == 1 and j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i] = conflict_points[index - 1]["down"].get(i, dict())
                                            
                                            conflict_points[index]["down"][i][12] = t12
                                            conflict_points[index]["down"][i][15] = t15
                                            conflict_points[index]["down"][i][18] = t18
                                        elif i == j == 1:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i - 1] = conflict_points[index - 1]["down"].get(i - 1, dict())
                                            
                                            conflict_points[index]["down"][i - 1][12] = t12
                                            conflict_points[index]["down"][i - 1][15] = t15
                                            conflict_points[index]["down"][i - 1][18] = t18
                                    elif left_turn[index][0] == 4:
                                        if i == j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9
                                        elif i == 1 and j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9
                                        elif i == j == 1:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i - 1] = conflict_points[index - 1]["down"].get(i - 1, dict())
                                            
                                            conflict_points[index - 1]["down"][i - 1][12] = t12
                                            conflict_points[index - 1]["down"][i - 1][15] = t15
                                            conflict_points[index - 1]["down"][i - 1][18] = t18
                                        elif i == 2 and j == 0:
                                            f_func, t_func = build_follow_function(s_func, segments)
                                        
                                            t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                            t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                            t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                            t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + arc_length + 1 * (j + 1), 0.0, 10.0)
                                            # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                            t18 = data[max(list(data.keys()))][-1].t1

                                            conflict_points[index] = conflict_points.get(index, dict())
                                            conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                            conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                            conflict_points[index]["left"][i][3] = t3
                                            conflict_points[index]["left"][i][6] = t6
                                            conflict_points[index]["left"][i][9] = t9

                                            conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                            conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                            conflict_points[index - 1]["down"][i - 1] = conflict_points[index - 1]["down"].get(i - 1, dict())
                                            
                                            conflict_points[index - 1]["down"][i - 1][12] = t12
                                            conflict_points[index - 1]["down"][i - 1][15] = t15
                                            conflict_points[index - 1]["down"][i - 1][18] = t18

                                    
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45
                                    line = Line(
                                        RIGHT * start_x,
                                        LEFT * 11.45
                                    ).shift(UP * (1.5 + 3 * i))

                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    points = {
                                        1: Point(0.1, 3.0),
                                        2: Point(1.0, 6.0),
                                        3: Point(1.0, 9.0),
                                        4: Point(1.0, 12.0),
                                        5: Point(1.0, 15.0),
                                        6: Point(1.0, 18.0)
                                    }

                                    # <-- использование кэш-обёртки вместо прямого func(...)
                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                        conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                        conflict_points[index]["left"][i][3] = t3
                                        conflict_points[index]["left"][i][6] = t6
                                        conflict_points[index]["left"][i][9] = t9
                                        conflict_points[index]["left"][i][12] = t12
                                        conflict_points[index]["left"][i][15] = t15
                                        conflict_points[index]["left"][i][18] = t18

                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t):
                                        def updater(mob):
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t

                                            s = s_func(t_local)
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                    else:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][0] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 1.5
                                        ).shift(UP * 1.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 1.5
                                        ).shift(UP * 4.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 4.5
                                        ).shift(UP * 4.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            RIGHT * start_x,
                                            RIGHT * 1.5
                                        ).shift(UP * 7.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            DOWN * 11.45
                                        ).shift(LEFT * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 0.0
                                    self.add(car)

                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой (начало дуги)
                                    p2 = np.array(line2.get_start())   # начало второй прямой (конец дуги)

                                    # центр четверти окружности (подходящий для перпендикулярных отрезков)
                                    center = np.array([p1[0], p2[1], 0.0])
                                    # радиус:
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle = atan2(p2[1] - center[1], p2[0] - center[0])

                                    # нормализуем span в диапазон [-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                    # --- используем cached_func / s_func как раньше ---
                                    points = {
                                        1: Point(conflict_points[index - 1]["up"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                        2: Point(conflict_points[index - 1]["up"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                        3: Point(conflict_points[index - 1]["up"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                        4: Point(conflict_points[index - 1]["down"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                        5: Point(conflict_points[index - 1]["down"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                        6: Point(conflict_points[index - 1]["down"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index),
                                    }
                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                        conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                        conflict_points[index]["left"][i][3] = t3
                                        conflict_points[index]["left"][i][6] = t6
                                        conflict_points[index]["left"][i][9] = t9
                                        conflict_points[index]["left"][i][12] = t12
                                        conflict_points[index]["left"][i][15] = t15
                                        conflict_points[index]["left"][i][18] = t18

                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7
                                    line = Line(
                                        RIGHT * start_x,
                                        LEFT * 11.45
                                    ).shift(UP * (1.5 + 3 * i))

                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'
                                    points = {
                                        1: Point(conflict_points[index - 1]["up"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                        2: Point(conflict_points[index - 1]["up"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                        3: Point(conflict_points[index - 1]["up"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                        4: Point(conflict_points[index - 1]["down"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                        5: Point(conflict_points[index - 1]["down"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                        6: Point(conflict_points[index - 1]["down"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index),
                                    }

                                    # print(index, direction, i, j)
                                    # print(points)
                                    # print()

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    
                                    
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["left"] = conflict_points[index].get('left', dict())
                                        conflict_points[index]["left"][i] = conflict_points[index]["left"].get(i, dict())

                                        conflict_points[index]["left"][i][3] = t3
                                        conflict_points[index]["left"][i][6] = t6
                                        conflict_points[index]["left"][i][9] = t9
                                        conflict_points[index]["left"][i][12] = t12
                                        conflict_points[index]["left"][i][15] = t15
                                        conflict_points[index]["left"][i][18] = t18

                                    
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                elif direction == "up":
                    if index == 0:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][1] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 1.5
                                        ).shift(RIGHT * 1.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 1.5
                                        ).shift(RIGHT * 4.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 4.5
                                        ).shift(RIGHT * 4.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 1.5
                                        ).shift(RIGHT * 7.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 1.57
                                    self.add(car)


                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой
                                    p2 = np.array(line2.get_start())   # начало второй прямой

                                    # центр дуги: x от начала второй линии (p2.x), y от конца первой линии (p1.y)
                                    center = np.array([p2[0], p1[1], 0.0])

                                    # радиус (магнитуда вектора от центра до p1)
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle   = atan2(p2[1] - center[1], p2[0] - center[0])


                                    # приводим разность углов к диапазону (-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    # длины сегментов
                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                   

                                    # --- используем cached_func / s_func как раньше ---
                                    points = {
                                        1: Point(conflict_points[index]["left"][0][9 - 3 * i], 9.0 + 1),
                                        2: Point(conflict_points[index]["left"][1][9 - 3 * i], 12.0 + 1),
                                        3: Point(conflict_points[index]["left"][2][9 - 3 * i], 15.0 + 1)
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["up"] = conflict_points[index].get('up', dict())
                                        conflict_points[index]["up"][i] = conflict_points[index]["up"].get(i, dict())

                                        conflict_points[index]["up"][i][3] = t3
                                        conflict_points[index]["up"][i][6] = t6
                                        conflict_points[index]["up"][i][9] = t9
                                        conflict_points[index]["up"][i][12] = t12
                                        conflict_points[index]["up"][i][15] = t15
                                        conflict_points[index]["up"][i][18] = t18
                                        
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45
                                    line = Line(
                                        DOWN * start_x,
                                        UP * 11.45
                                    ).shift(RIGHT * (1.5 + 3 * i))

                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'
                                    points = {
                                        1: Point(conflict_points[index]["left"][0][9 - 3 * i], 9.0 + 1),
                                        2: Point(conflict_points[index]["left"][1][9 - 3 * i], 12.0 + 1),
                                        3: Point(conflict_points[index]["left"][2][9 - 3 * i], 15.0 + 1)
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))

                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["up"] = conflict_points[index].get('up', dict())
                                        conflict_points[index]["up"][i] = conflict_points[index]["up"].get(i, dict())

                                        conflict_points[index]["up"][i][3] = t3
                                        conflict_points[index]["up"][i][6] = t6
                                        conflict_points[index]["up"][i][9] = t9
                                        conflict_points[index]["up"][i][12] = t12
                                        conflict_points[index]["up"][i][15] = t15
                                        conflict_points[index]["up"][i][18] = t18

                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                    else:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][1] >= 2 ** i + j:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 1.5
                                        ).shift(RIGHT * 1.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 1.5
                                        ).shift(RIGHT * 4.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 4.5
                                        ).shift(RIGHT * 4.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            DOWN * start_x,
                                            DOWN * 1.5
                                        ).shift(RIGHT * 7.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            LEFT * 11.45
                                        ).shift(UP * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 1.57
                                    self.add(car)

                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой
                                    p2 = np.array(line2.get_start())   # начало второй прямой

                                    # центр четверти окружности
                                    center = np.array([p2[0], p1[1], 0.0])

                                    # радиус
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle   = atan2(p2[1] - center[1], p2[0] - center[0])

                                    # нормализуем span в диапазон [-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                    # --- используем cached_func / s_func как раньше ---
                                    points = {
                                        1: Point(conflict_points[index - 1]["right"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                        2: Point(conflict_points[index - 1]["right"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                        3: Point(conflict_points[index - 1]["right"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                        4: Point(conflict_points[index]["left"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                        5: Point(conflict_points[index]["left"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                        6: Point(conflict_points[index]["left"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index)
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["up"] = conflict_points[index].get('up', dict())
                                        conflict_points[index]["up"][i] = conflict_points[index]["up"].get(i, dict())

                                        conflict_points[index]["up"][i][3] = t3
                                        conflict_points[index]["up"][i][6] = t6
                                        conflict_points[index]["up"][i][9] = t9
                                        conflict_points[index]["up"][i][12] = t12
                                        conflict_points[index]["up"][i][15] = t15
                                        conflict_points[index]["up"][i][18] = t18
                                        
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:

                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7
                                    line = Line(
                                        DOWN * start_x,
                                        UP * 11.45
                                    ).shift(RIGHT * (1.5 + 3 * i))

                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'
                                    points = {
                                        1: Point(conflict_points[index - 1]["right"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                        2: Point(conflict_points[index - 1]["right"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                        3: Point(conflict_points[index - 1]["right"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                        4: Point(conflict_points[index]["left"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                        5: Point(conflict_points[index]["left"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                        6: Point(conflict_points[index]["left"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index)
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["up"] = conflict_points[index].get('up', dict())
                                        conflict_points[index]["up"][i] = conflict_points[index]["up"].get(i, dict())

                                        conflict_points[index]["up"][i][3] = t3
                                        conflict_points[index]["up"][i][6] = t6
                                        conflict_points[index]["up"][i][9] = t9
                                        conflict_points[index]["up"][i][12] = t12
                                        conflict_points[index]["up"][i][15] = t15
                                        conflict_points[index]["up"][i][18] = t18

                                    
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                elif direction == "right":
                    if index == 0:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][2] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 1.5
                                        ).shift(DOWN * 1.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 1.5
                                        ).shift(DOWN * 4.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 4.5
                                        ).shift(DOWN * 4.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 1.5
                                        ).shift(DOWN * 7.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 3.14
                                    self.add(car)


                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой
                                    p2 = np.array(line2.get_start())   # начало второй прямой

                                    # центр дуги: x от начала второй линии (p2.x), y от конца первой линии (p1.y)
                                    center = np.array([p1[0], p2[1], 0.0])

                                    # радиус (магнитуда вектора от центра до p1)
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle   = atan2(p2[1] - center[1], p2[0] - center[0])


                                    # приводим разность углов к диапазону (-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    # длины сегментов
                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                   

                                    # --- используем cached_func / s_func как раньше ---
                                    points = dict()
                                    conflict_points[index - 1] = conflict_points.get(index - 1, dict())
                                    conflict_points[index - 1]["down"] = conflict_points[index - 1].get('down', dict())
                                    conflict_points[index - 1]["down"][0] = conflict_points[index - 1]['down'].get(0, 0)
                                    conflict_points[index - 1]["down"][1] = conflict_points[index - 1]['down'].get(1, 0)

                                    if not conflict_points[index - 1]["down"]:
                                        points = {
                                            1: Point(conflict_points[index]["up"][0][9 - 3 * i], 9.0 + 1),
                                            2: Point(conflict_points[index]["up"][1][9 - 3 * i], 12.0 + 1),
                                            3: Point(conflict_points[index]["up"][2][9 - 3 * i], 15.0 + 1)
                                        }
                                    elif not conflict_points[index - 1]["down"][1]:
                                        points = {
                                            1: Point(conflict_points[index - 1]["down"][0][12 + 3 * i], 6.0 + 1),
                                            2: Point(conflict_points[index]["up"][0][9 - 3 * i], 9.0 + 1),
                                            3: Point(conflict_points[index]["up"][1][9 - 3 * i], 12.0 + 1),
                                            4: Point(conflict_points[index]["up"][2][9 - 3 * i], 15.0 + 1)
                                        }
                                    else:
                                        points = {
                                            1: Point(conflict_points[index - 1]["down"][1][12 + 3 * i], 3.0 + 1),
                                            2: Point(conflict_points[index - 1]["down"][0][12 + 3 * i], 6.0 + 1),
                                            3: Point(conflict_points[index]["up"][0][9 - 3 * i], 9.0 + 1),
                                            4: Point(conflict_points[index]["up"][1][9 - 3 * i], 12.0 + 1),
                                            5: Point(conflict_points[index]["up"][2][9 - 3 * i], 15.0 + 1)
                                        }
                                        print(points)
                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["right"] = conflict_points[index].get('right', dict())
                                        conflict_points[index]["right"][i] = conflict_points[index]["right"].get(i, dict())

                                        conflict_points[index]["right"][i][3] = t3
                                        conflict_points[index]["right"][i][6] = t6
                                        conflict_points[index]["right"][i][9] = t9
                                        conflict_points[index]["right"][i][12] = t12
                                        conflict_points[index]["right"][i][15] = t15
                                        conflict_points[index]["right"][i][18] = t18
                                        
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45
                                    line = Line(
                                        LEFT * start_x,
                                        RIGHT * 11.45
                                    ).shift(DOWN * (1.5 + 3 * i))

                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'
                                    points = {
                                        1: Point(conflict_points[index]["up"][0][9 - 3 * i], 9.0 + 1),
                                        2: Point(conflict_points[index]["up"][1][9 - 3 * i], 12.0 + 1),
                                        3: Point(conflict_points[index]["up"][2][9 - 3 * i], 15.0 + 1)
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))

                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["right"] = conflict_points[index].get('right', dict())
                                        conflict_points[index]["right"][i] = conflict_points[index]["right"].get(i, dict())

                                        conflict_points[index]["right"][i][3] = t3
                                        conflict_points[index]["right"][i][6] = t6
                                        conflict_points[index]["right"][i][9] = t9
                                        conflict_points[index]["right"][i][12] = t12
                                        conflict_points[index]["right"][i][15] = t15
                                        conflict_points[index]["right"][i][18] = t18

                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                    else:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][2] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 1.5
                                        ).shift(DOWN * 1.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 1.5
                                        ).shift(DOWN * 4.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 4.5
                                        ).shift(DOWN * 4.5)

                                        line2 = Line(
                                            UP * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            LEFT * start_x,
                                            LEFT * 1.5
                                        ).shift(DOWN * 7.5)

                                        line2 = Line(
                                            DOWN * 1.5,
                                            UP * 11.45
                                        ).shift(RIGHT * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 3.14
                                    self.add(car)


                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой
                                    p2 = np.array(line2.get_start())   # начало второй прямой

                                    # центр дуги: x от начала второй линии (p2.x), y от конца первой линии (p1.y)
                                    center = np.array([p1[0], p2[1], 0.0])

                                    # радиус (магнитуда вектора от центра до p1)
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle   = atan2(p2[1] - center[1], p2[0] - center[0])


                                    # приводим разность углов к диапазону (-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    # длины сегментов
                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                   

                                    # --- используем cached_func / s_func как раньше ---
                                    if i == 1 and index == 2: 
                                        points = {
                                            1: Point(conflict_points[index - 1]["right"][2][12 + 3 * i] + 1.4, 0.0 + 1 + 17.7 * index),
                                            2: Point(conflict_points[index - 1]["right"][1][12 + 3 * i] + 1.4, 3.0 + 1 + 17.7 * index),
                                            3: Point(conflict_points[index - 1]["right"][0][12 + 3 * i] + 1.4, 6.0 + 1 + 17.7 * index),
                                            4: Point(conflict_points[index]["left"][0][9 - 3 * i] + 1.4, 9.0 + 1 + 17.7 * index),
                                            5: Point(conflict_points[index]["left"][1][9 - 3 * i] + 1.4, 12.0 + 1 + 17.7 * index),
                                            6: Point(conflict_points[index]["left"][2][9 - 3 * i] + 1.4, 15.0 + 1 + 17.7 * index),
                                        }
                                    else:
                                        points = {
                                            1: Point(conflict_points[index - 1]["down"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                            2: Point(conflict_points[index - 1]["down"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                            3: Point(conflict_points[index - 1]["down"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                            4: Point(conflict_points[index]["up"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                            5: Point(conflict_points[index]["up"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                            6: Point(conflict_points[index]["up"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index),
                                        }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["right"] = conflict_points[index].get('right', dict())
                                        conflict_points[index]["right"][i] = conflict_points[index]["right"].get(i, dict())

                                        conflict_points[index]["right"][i][3] = t3
                                        conflict_points[index]["right"][i][6] = t6
                                        conflict_points[index]["right"][i][9] = t9
                                        conflict_points[index]["right"][i][12] = t12
                                        conflict_points[index]["right"][i][15] = t15
                                        conflict_points[index]["right"][i][18] = t18
                                        
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7
                                    line = Line(
                                        LEFT * start_x,
                                        RIGHT * 11.45
                                    ).shift(DOWN * (1.5 + 3 * i))

                                    car = Rectangle(width=4.9, height=1.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'

                                    if i == 1 and index == 2: 
                                        points = {
                                            1: Point(conflict_points[index - 1]["right"][2][12 + 3 * i] + 1.4, 0.0 + 1 + 17.7 * index),
                                            2: Point(conflict_points[index - 1]["right"][1][12 + 3 * i] + 1.4, 3.0 + 1 + 17.7 * index),
                                            3: Point(conflict_points[index - 1]["right"][0][12 + 3 * i] + 1.4, 6.0 + 1 + 17.7 * index),
                                            4: Point(conflict_points[index]["left"][0][9 - 3 * i] + 1.4, 9.0 + 1 + 17.7 * index),
                                            5: Point(conflict_points[index]["left"][1][9 - 3 * i] + 1.4, 12.0 + 1 + 17.7 * index),
                                            6: Point(conflict_points[index]["left"][2][9 - 3 * i] + 1.4, 15.0 + 1 + 17.7 * index),
                                        }
                                    else:
                                        points = {
                                            1: Point(conflict_points[index - 1]["down"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                            2: Point(conflict_points[index - 1]["down"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                            3: Point(conflict_points[index - 1]["down"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                            4: Point(conflict_points[index]["up"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                            5: Point(conflict_points[index]["up"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                            6: Point(conflict_points[index]["up"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index),
                                        }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["right"] = conflict_points[index].get('right', dict())
                                        conflict_points[index]["right"][i] = conflict_points[index]["right"].get(i, dict())

                                        conflict_points[index]["right"][i][3] = t3
                                        conflict_points[index]["right"][i][6] = t6
                                        conflict_points[index]["right"][i][9] = t9
                                        conflict_points[index]["right"][i][12] = t12
                                        conflict_points[index]["right"][i][15] = t15
                                        conflict_points[index]["right"][i][18] = t18

                                    
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                elif direction == "down":
                    if index == 0:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][3] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 1.5
                                        ).shift(LEFT * 1.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 1.5
                                        ).shift(LEFT * 4.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 4.5
                                        ).shift(LEFT * 4.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 1.5
                                        ).shift(LEFT * 7.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 4.5)

                                    
                                    # --- машина ---
                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 1.57 
                                    self.add(car)


                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой
                                    p2 = np.array(line2.get_start())   # начало второй прямой

                                    # центр дуги: x от начала второй линии (p2.x), y от конца первой линии (p1.y)
                                    center = np.array([p2[0], p1[1], 0.0])

                                    # радиус (магнитуда вектора от центра до p1)
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle   = atan2(p2[1] - center[1], p2[0] - center[0])


                                    # приводим разность углов к диапазону (-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    # длины сегментов
                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                   

                                    # --- используем cached_func / s_func как раньше ---
                                    points = {
                                        1: Point(conflict_points[index]["left"][2][12 + 3 * i], 0.0 + 1),
                                        # 2: Point(conflict_points[index]["left"][1][12 + 3 * i], 3.0 + 1),
                                        # 3: Point(conflict_points[index]["left"][0][12 + 3 * i], 6.0 + 1),
                                        2: Point(conflict_points[index]["right"][0][9 - 3 * i], 9.0 + 1),
                                        3: Point(conflict_points[index]["right"][1][9 - 3 * i], 12.0 + 1),
                                        4: Point(conflict_points[index]["right"][2][9 - 3 * i], 15.0 + 1),
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["down"] = conflict_points[index].get('down', dict())
                                        conflict_points[index]["down"][i] = conflict_points[index]["down"].get(i, dict())

                                        conflict_points[index]["down"][i][3] = t3
                                        conflict_points[index]["down"][i][6] = t6
                                        conflict_points[index]["down"][i][9] = t9
                                        conflict_points[index]["down"][i][12] = t12
                                        conflict_points[index]["down"][i][15] = t15
                                        conflict_points[index]["down"][i][18] = t18
                                        
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45
                                    line = Line(
                                        UP * start_x,
                                        DOWN * 11.45
                                    ).shift(LEFT * (1.5 + 3 * i))

                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'
                                    points = {
                                        1: Point(conflict_points[index]["left"][2][12 + 3 * i], 0.0 + 1),
                                        # 2: Point(conflict_points[index]["left"][1][12 + 3 * i], 3.0 + 1),
                                        # 3: Point(conflict_points[index]["left"][0][12 + 3 * i], 6.0 + 1),
                                        2: Point(conflict_points[index]["right"][0][9 - 3 * i], 9.0 + 1),
                                        3: Point(conflict_points[index]["right"][1][9 - 3 * i], 12.0 + 1),
                                        4: Point(conflict_points[index]["right"][2][9 - 3 * i], 15.0 + 1),
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1))
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1), 0.0, 10.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1), 0.0, 10.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1), 0.0, 10.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1), 0.0, 10.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1), 0.0, 10.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1), 0.0, 10.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["down"] = conflict_points[index].get('down', dict())
                                        conflict_points[index]["down"][i] = conflict_points[index]["down"].get(i, dict())

                                        conflict_points[index]["down"][i][3] = t3
                                        conflict_points[index]["down"][i][6] = t6
                                        conflict_points[index]["down"][i][9] = t9
                                        conflict_points[index]["down"][i][12] = t12
                                        conflict_points[index]["down"][i][15] = t15
                                        conflict_points[index]["down"][i][18] = t18

                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )
                    else:
                        for i in range(3):
                            for j in range(i + 1):
                                if left_turn[index][3] >= 2 ** i + j:
                                    # --- строим прямые как раньше ---
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7

                                    if i == 0 and j == 0:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 1.5
                                        ).shift(LEFT * 1.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 1.5)

                                    elif i == 1 and j == 0:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 1.5
                                        ).shift(LEFT * 4.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 4.5)

                                    elif i == 1 and j == 1:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 4.5
                                        ).shift(LEFT * 4.5)

                                        line2 = Line(
                                            RIGHT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 1.5)

                                    elif i == 2 and j == 0:
                                        line1 = Line(
                                            UP * start_x,
                                            UP * 1.5
                                        ).shift(LEFT * 7.5)

                                        line2 = Line(
                                            LEFT * 1.5,
                                            RIGHT * 11.45
                                        ).shift(DOWN * 4.5)

                                    # --- машина ---
                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line1.point_from_proportion(0))
                                    # сохраняем начальный "предыдущий" угол для аккуратного обновления поворота
                                    car._prev_angle = 1.57 
                                    self.add(car)


                                    # --- вычисляем параметры дуги между концом line1 и началом line2 ---
                                    p1 = np.array(line1.get_end())     # конец первой прямой
                                    p2 = np.array(line2.get_start())   # начало второй прямой

                                    # центр дуги: x от начала второй линии (p2.x), y от конца первой линии (p1.y)
                                    center = np.array([p2[0], p1[1], 0.0])

                                    # радиус (магнитуда вектора от центра до p1)
                                    radius = float(np.linalg.norm(p1 - center))

                                    # углы начала и конца дуги (в радианах)
                                    start_angle = atan2(p1[1] - center[1], p1[0] - center[0])
                                    end_angle   = atan2(p2[1] - center[1], p2[0] - center[0])


                                    # приводим разность углов к диапазону (-pi, pi]
                                    angle_span = end_angle - start_angle
                                    if angle_span <= -pi:
                                        angle_span += 2 * pi
                                    elif angle_span > pi:
                                        angle_span -= 2 * pi

                                    # длины сегментов
                                    arc_length = abs(radius * angle_span)
                                    line1_length = float(line1.get_length())
                                    line2_length = float(line2.get_length())
                                    total_path_length = line1_length + arc_length + line2_length

                                   

                                    # --- используем cached_func / s_func как раньше ---
                                    points = {
                                        1: Point(conflict_points[index]["left"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                        2: Point(conflict_points[index]["left"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                        3: Point(conflict_points[index]["left"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                        4: Point(conflict_points[index]["right"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                        5: Point(conflict_points[index]["right"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                        6: Point(conflict_points[index]["right"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index),
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["down"] = conflict_points[index].get('down', dict())
                                        conflict_points[index]["down"][i] = conflict_points[index]["down"].get(i, dict())

                                        conflict_points[index]["down"][i][3] = t3
                                        conflict_points[index]["down"][i][6] = t6
                                        conflict_points[index]["down"][i][9] = t9
                                        conflict_points[index]["down"][i][12] = t12
                                        conflict_points[index]["down"][i][15] = t15
                                        conflict_points[index]["down"][i][18] = t18
                                        
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1
                                    total_time = max(total_time, car_total_t)

                                    # ---------- исправленный апдейтер: фиксируем все параметры в замыкании ----------
                                    def make_updater_for_turn(
                                        s_func, s_max, car_total_t,
                                        line1, line2,
                                        center, radius,
                                        start_angle, angle_span,
                                        line1_length, line2_length,
                                        arc_length, total_path_length
                                    ):
                                        """
                                        Все необходимые параметры передаются как аргументы — каждая машина
                                        получит свою «зафиксированную» копию траектории.
                                        """

                                        # функция положения на объединённом пути по абсолютному расстоянию s (0..total_path_length)
                                        def pos_on_path(s):
                                            if s <= 0:
                                                return np.array(line1.get_start())
                                            if s < line1_length:
                                                return np.array(line1.point_from_proportion(s / line1_length))
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                return center + np.array([radius * cos(theta), radius * sin(theta), 0.0])
                                            # else — вторая прямая
                                            s_rem2 = s_rem - arc_length
                                            if s_rem2 >= line2_length:
                                                return np.array(line2.get_end())
                                            return np.array(line2.point_from_proportion(s_rem2 / line2_length))

                                        # функция тангенса (угла) в точке s — направление движения
                                        def tangent_angle_on_path(s):
                                            eps = 1e-6
                                            if s <= 0:
                                                p0 = np.array(line1.get_start())
                                                p1_ = np.array(line1.get_end())
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            if s < line1_length:
                                                p0 = np.array(line1.point_from_proportion(max(0, (s - eps) / line1_length)))
                                                p1_ = np.array(line1.point_from_proportion(min(1, (s + eps) / line1_length)))
                                                dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                                return atan2(dy, dx)
                                            s_rem = s - line1_length
                                            if s_rem < arc_length:
                                                frac = s_rem / arc_length
                                                theta = start_angle + frac * angle_span
                                                # для круга касательный угол = theta + pi/2 (с учётом направления)
                                                return theta + (pi / 2.0 if angle_span >= 0 else -pi / 2.0)
                                            s_rem2 = s_rem - arc_length
                                            p0 = np.array(line2.point_from_proportion(max(0, (s_rem2 - eps) / line2_length)))
                                            p1_ = np.array(line2.point_from_proportion(min(1, (s_rem2 + eps) / line2_length)))
                                            dx, dy = p1_[0] - p0[0], p1_[1] - p0[1]
                                            return atan2(dy, dx)

                                        def updater(mob):
                                            t = tracker.get_value()
                                            t_local = car_total_t if t > car_total_t else t

                                            s = s_func(t_local)               # модельный пройденный путь
                                            # приводим к абсолютной длине визуального пути
                                            alpha = s / s_max if s_max != 0 else 0.0
                                            s_abs = alpha * total_path_length

                                            pos = pos_on_path(s_abs)
                                            mob.move_to(pos)

                                            desired_angle = tangent_angle_on_path(s_abs)
                                            # нормализуем угол в -pi..pi
                                            while desired_angle <= -pi:
                                                desired_angle += 2 * pi
                                            while desired_angle > pi:
                                                desired_angle -= 2 * pi

                                            prev = getattr(mob, "_prev_angle", 0.0)
                                            # корректируем разницу (берём ближайшую дельту через 2pi)
                                            delta = desired_angle - prev
                                            if delta > pi:
                                                delta -= 2 * pi
                                            elif delta < -pi:
                                                delta += 2 * pi

                                            # вращаем относительно центра объекта (чтобы не смещать)
                                            mob.rotate(delta, about_point=mob.get_center())
                                            mob._prev_angle = desired_angle

                                        return updater

                                    # При создании апдейтера — явно передаём все параметры, чтобы зафиксировать их
                                    car.add_updater(
                                        make_updater_for_turn(
                                            s_func, s_max, car_total_t,
                                            line1, line2,
                                            center, radius,
                                            start_angle, angle_span,
                                            line1_length, line2_length,
                                            arc_length, total_path_length
                                        )
                                    )
                                else:
                                    start_x = 9 + 1 * (j + 1) + 4.9 * j + 2.45 + index * 17.7
                                    line = Line(
                                        UP * start_x,
                                        DOWN * 11.45
                                    ).shift(LEFT * (1.5 + 3 * i))

                                    car = Rectangle(width=1.9, height=4.9, color=YELLOW)
                                    car.move_to(line.point_from_proportion(0))
                                    self.add(car)

                                    # используем времена конфликтов, найденные для 'left'
                                    points = {
                                        1: Point(conflict_points[index]["left"][2][12 + 3 * i], 0.0 + 1 + 17.7 * index),
                                        2: Point(conflict_points[index]["left"][1][12 + 3 * i], 3.0 + 1 + 17.7 * index),
                                        3: Point(conflict_points[index]["left"][0][12 + 3 * i], 6.0 + 1 + 17.7 * index),
                                        4: Point(conflict_points[index]["right"][0][9 - 3 * i], 9.0 + 1 + 17.7 * index),
                                        5: Point(conflict_points[index]["right"][1][9 - 3 * i], 12.0 + 1 + 17.7 * index),
                                        6: Point(conflict_points[index]["right"][2][9 - 3 * i], 15.0 + 1 + 17.7 * index),
                                    }

                                    data = cached_func(points, 18 + 1 * (j + 1) + 4.9 * (j + 1) + index * 17.7)
                                    
                                    s_func, segments = build_position_function(data)

                                    if i == j:
                                        f_func, t_func = build_follow_function(s_func, segments)
                                        
                                        t3 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 1 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t6 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 2 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t9 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 3 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t12 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 4 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t15 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 5 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        # t18 = invert_monotonic_function(f_func, 4.9 * (j + 1) + 3 * 6 + 1 * (j + 1) + index * 17.7, 0.0, 10.0 + index * 12.0)
                                        t18 = data[max(list(data.keys()))][-1].t1

                                        conflict_points[index] = conflict_points.get(index, dict())
                                        conflict_points[index]["down"] = conflict_points[index].get('down', dict())
                                        conflict_points[index]["down"][i] = conflict_points[index]["down"].get(i, dict())

                                        conflict_points[index]["down"][i][3] = t3
                                        conflict_points[index]["down"][i][6] = t6
                                        conflict_points[index]["down"][i][9] = t9
                                        conflict_points[index]["down"][i][12] = t12
                                        conflict_points[index]["down"][i][15] = t15
                                        conflict_points[index]["down"][i][18] = t18

                                    
                                    car_total_t = segments[-1].t1
                                    s_max = segments[-1].s1

                                    total_time = max(total_time, car_total_t)

                                    # ---------- правильный updater ----------
                                    def make_updater(line, s_func, s_max, car_total_t, i=i, j=j):
                                        def updater(mob):
                                            
                                            t = tracker.get_value()

                                            # если машина уже доехала — больше не двигаем
                                            if t > car_total_t:
                                                t_local = car_total_t
                                            else:
                                                t_local = t
                                            
                                            s = s_func(t_local)
                                            
                                            alpha = s / s_max if s_max != 0 else 0
                                            pos = line.point_from_proportion(alpha)
                                            mob.move_to(pos)

                                        return updater

                                    car.add_updater(
                                        make_updater(line, s_func, s_max, car_total_t)
                                    )



        self.wait(1)

        # Запуск анимации: tracker управляет временем, таймер привязан к tracker'у
        self.play(
            tracker.animate.set_value(total_time),
            run_time=total_time,
            rate_func=linear
        )

        self.wait(2)