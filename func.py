from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple
import math

# -----------------------------
# Структуры
# -----------------------------
@dataclass
class Segment:
    t0: float
    t1: float
    nameAccel: str
    a0: float
    a1: float
    v0: float
    v1: float
    s0: float
    safety_s: float
    s1: float
    s_free: float
    sTarget: float

@dataclass
class Point:
    t: float
    s: float

# -----------------------------
# Утилиты численных методов
# -----------------------------
def _composite_simpson(func: Callable[[float], float], a: float, b: float, n: int = 200) -> float:
    """
    Композитная формула Симпсона на N отрезках (n должно быть чётным).
    По умолчанию n=200 (как в оригинальном C++ в одном месте).
    """
    if a == b:
        return 0.0
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        x = a + i * h
        s += (4.0 if i % 2 == 1 else 2.0) * func(x)
    return s * (h / 3.0)

def _bisection_root(func: Callable[[float], float], lo: float, hi: float, tol: float = 1e-9, max_iter: int = 50) -> float:
    """
    Простая бисекция — возвращаем найденный корень в середине последнего интервала.
    Требует, чтобы func(lo) и func(hi) имели разные знаки (или один из них == 0).
    """
    flo = func(lo)
    fhi = func(hi)
    if math.isnan(flo) or math.isnan(fhi):
        return float('nan')
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        # нет гарантии смены знака
        return float('nan')

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = func(mid)
        if abs(fmid) <= tol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)

# -----------------------------
# Основные функции (перевод C++ логики)
# -----------------------------
def display(m: Dict[int, List[Segment]], i: int) -> None:
    """
    Печать сегментов для ключа i в формате, похожем на C++ вывод.
    """
    if i not in m:
        print(f"No segments for key {i}")
        return
    for seg in m[i]:
        print("  Interval: [", str(seg.t0), ",", str(seg.t1), "] -", str(seg.nameAccel) + ",",
              "a0 =", str(seg.a0) + ",", "a1 =", str(seg.a1) + ",",
              "v0 =", str(seg.v0) + ",", "v1 =", str(seg.v1) + ",",
              "s0 =", str(seg.s0) + ",", "safety_s =", str(seg.safety_s) + ",",
              "s1 =", str(seg.s1) + ",", "s_free =", str(seg.s_free) + ",",
              "sTarget =", str(seg.sTarget))

def acceleration(t1: float, a0: float, t0: float, k: float) -> float:
    """
    Перевод C++ acceleration(...)
    """
    if t1 <= t0:
        if k == 1.0:
            return max(2.0, a0)
        elif k == -1.0:
            return min(-2.0, a0)
        else:
            return 0.0

    if k > 0.0:
        return min(max(2.0, a0) + k * (t1 - t0), 4.0)
    return max(min(-2.0, a0) + k * (t1 - t0), -4.0)

def velocity(t1: float, a0: float, v0: float, t0: float, k: float) -> float:
    """
    Перевод C++ velocity(...) с заменой gauss_kronrod на composite Simpson.
    Сохраняем структуру ветвления как в C++:
      - k == 1.0 : увеличение ускорения (aIncrease)
      - k == 0.0 : возвращаем v0 (константа)
      - иначе   : уменьшение ускорения (aDecrease)
    """
    if t1 <= t0:
        return v0

    # локальные lambda-аналогичные C++ aIncrease / aDecrease
    def a_increase(t: float) -> float:
        return min(max(2.0, a0) + 1.0 * (t - t0), 4.0)

    def a_decrease(t: float) -> float:
        # соответствует C++: min(-2.0, a0) - 1.0 * (t - t0)
        return min(-2.0, a0) - 1.0 * (t - t0)

    dt_total = t1 - t0

    if k == 1.0:
        tAmax = t0 + 4.0 - max(a0, 2.0)  # время, когда accel достигает 4
        if (tAmax - t0) >= dt_total:
            # интегрируем aIncrease на [t0,t1]
            integral = _composite_simpson(a_increase, t0, t1, n=200)
            return min(v0 + integral, 17.0)
        elif a0 == 4.0:
            integral = 4.0 * (t1 - t0)
            return min(v0 + integral, 17.0)
        else:
            integral1 = _composite_simpson(a_increase, t0, tAmax, n=200)
            integral2 = 4.0 * (t1 - tAmax)
            return min(v0 + integral1 + integral2, 17.0)

    elif k == 0.0:
        return v0

    else:
        # k != 1.0 and k != 0.0 => уменьшение
        tAmax = t0 + 4.0 + min(a0, -2.0)
        if (tAmax - t0) >= dt_total:
            integral = _composite_simpson(a_decrease, t0, t1, n=200)
            return min(v0 + integral, 17.0)
        elif a0 == 4.0:
            integral = 4.0 * (t1 - t0)
            return min(v0 + integral, 17.0)
        else:
            integral1 = _composite_simpson(a_decrease, t0, tAmax, n=200)
            integral2 = 4.0 * (t1 - tAmax)
            return min(v0 + integral1 + integral2, 17.0)

def distance(t1: float, a0: float, v0: float, s0: float, t0: float, k: float) -> float:
    """
    Перевод C++ distance(...) через интеграл от velocity.
    """
    if t1 <= t0:
        return s0

    def v_func(t: float) -> float:
        return velocity(t, a0, v0, t0, k)

    integral = _composite_simpson(v_func, t0, t1, n=200)
    return s0 + integral

def safety_distance(t1: float, a0: float, v0: float, t0: float, k: float) -> float:
    """
    Перевод C++ safety_distance(...)
    """
    s = 0.0
    v = velocity(t1, a0, v0, t0, k)
    a = min(acceleration(t1, a0, t0, k), -2.0)

    # Фаза 1: линейное падение ускорения до -4 (k = -1 логика)
    if a > -4.0:
        # dt — время до достижения -4 если k = -1 (в C++: dt = (4.0 + a))
        dt = (4.0 + a)  # поскольку a отрицательное
        # v_end после этой фазы (встроенное интегрирование ускорения: v + a*dt + 0.5*(-1)*dt^2)
        v_end = v + a * dt + 0.5 * (-1.0) * dt * dt

        # Если остановимся до достижения -4
        if v_end <= 0.0:
            # решаем квадратное уравнение для времени остановки:
            # v(t) = v + a*t + 0.5*(-1)*t^2 = 0 -> -0.5 t^2 + a t + v = 0
            A = -0.5
            B = a
            C = v
            D = B * B - 4.0 * A * C
            if D < 0.0:
                # численная аномалия — возвращаем пройденный путь по аппроксимации
                # (в оригинале это бы вызвало sqrt отрицательного)
                return s
            t_stop = (-B - math.sqrt(D)) / (2.0 * A)
            # пройденный путь до остановки: интеграл v(t) dt
            s = v * t_stop + 0.5 * a * t_stop * t_stop + (1.0 / 6.0) * (-1.0) * t_stop * t_stop * t_stop
            return s

        # иначе добавляем путь этой фазы
        s += v * dt + 0.5 * a * dt * dt + (1.0 / 6.0) * (-1.0) * dt * dt * dt
        v = v_end
        a = -4.0

    # Фаза 2: постоянное ускорение -4 -> путь торможения v^2/(2*|a|) где |a| = 4 -> (v^2)/8
    s += (v * v) / 8.0
    return s

# -----------------------------
# Пересечения и проверка положения
# -----------------------------
def count_intersections(t2: float, s2: float, slope: float, t_min: float, t_max: float,
                        a0: float, v0: float, s0: float, t0: float, k: float) -> int:
    """
    Аналог C++ countIntersections(...).
    Скользим по [t_min,t_max] с N шагами (N=200), ищем смены знака F(t) = s_curve(t) - s_line(t)
    При смене знака используем бисекцию, затем считаем корень, если root > t2 -> увеличиваем счетчик.
    """
    def F(t: float) -> float:
        s_curve = distance(t, a0, v0, s0, t0, k)
        s_line = s2 + slope * (t - t2)
        return s_curve - s_line

    count = 0
    N = 200
    if t_max <= t_min:
        return 0
    dt = (t_max - t_min) / N

    prev_t = t_min
    prev_F = F(prev_t)

    for i in range(1, N + 1):
        t = t_min + i * dt
        val = F(t)

        # если предыдущая точка имеет значение 0 — учитываем пересечение
        if prev_F == 0.0:
            count += 1
        elif prev_F * val < 0.0:
            # есть смена знака — найдем корень на интервале (prev_t, t)
            root = _bisection_root(F, prev_t, t, tol=1e-9, max_iter=50)
            if not math.isnan(root) and root > t2:
                count += 1

        prev_t = t
        prev_F = val

    return count

def is_above_line(t0_: float, s0_: float, slope: float, t: float, s: float) -> int:
    """
    Возвращает 1 если точка (t,s) расположена выше линии s_line = s0 + slope*(t - t0)
    """
    s_line = s0_ + slope * (t - t0_)
    return 1 if s > s_line else 0

# --------------------------------------------
# t_for_s_on_line
# --------------------------------------------
def t_for_s_on_line(s_target: float, s2: float, t2: float, slope: float) -> float:
    """
    Решает: s_target = s2 + slope * (t - t2)
    """
    if abs(slope) < 1e-12:
        raise RuntimeError("Slope is zero — cannot solve for t.")

    return t2 + (s_target - s2) / slope

# --------------------------------------------
# t_for_stop
# --------------------------------------------
def t_for_stop(t1: float, a0: float, v0: float, t0: float, k: float) -> float:
    # Уже остановлена
    if v0 <= 0:
        return 0.0
    
    c = max(min(-2.0, a0), -4.0)

    # Если уже постоянное -4
    if c == -4.0:
        return v0 / 4.0

    tAMin = 4.0 + c
    v_switch = velocity(t0 + tAMin, a0, v0, t0, -1.0)

    # ---------- Фаза 1 ----------
    if v_switch <= 0.0:

        D = c*c - 4.0*v0

        if D < 0.0:
            return 0.0  # чтобы не вернуть None

        sqrtD = math.sqrt(D)

        t_root1 = (c + sqrtD) / 2.0
        t_root2 = (c - sqrtD) / 2.0

        candidates = [
            t for t in (t_root1, t_root2)
            if 0.0 <= t <= tAMin
        ]

        if not candidates:
            return 0.0

        return min(candidates)

    # ---------- Фаза 2 ----------
    return tAMin + v_switch / 4.0

# --------------------------------------------
# t_for_s_on_curve
# --------------------------------------------
def t_for_s_on_curve(s_target: float,
                     t_min: float,
                     t_max: float,
                     a0: float,
                     v0: float,
                     s0: float,
                     t0: float,
                     k: float) -> float:
    """
    Решает distance(t, ...) = s_target
    """
    def F(t: float) -> float:
        return distance(t, a0, v0, s0, t0, k) - s_target

    f_min = F(t_min)
    f_max = F(t_max)

    if f_min * f_max > 0:
        raise RuntimeError("t_for_s_on_curve: root is not bracketed!")

    root = _bisection_root(F, t_min, t_max, tol=1e-12, max_iter=50)

    if math.isnan(root):
        raise RuntimeError("Root finding failed")

    return root


# --------------------------------------------
# safeFindRoot
# --------------------------------------------
def safe_find_root(g, t0: float, t1: float) -> float:
    """
    Безопасный поиск корня. Возвращает NaN при ошибке.
    """
    try:
        root = _bisection_root(g, t0, t1, tol=1e-12, max_iter=50)
        return root
    except Exception:
        return float("nan")


# --------------------------------------------
# Простые аналитические формулы (1:1 перенос)
# --------------------------------------------
def sMax(t: float, v0: float, s0: float) -> float:
    return 2.0 * t * t + v0 * t + s0

def vMax(t: float, v0: float) -> float:
    return 4.0 * t + v0

def sIncrease(t: float, c: float, v0: float, s0: float) -> float:
    return (1.0 / 6.0) * t**3 + 0.5 * c * t**2 + v0 * t + s0

def vIncrease(t: float, c: float, v0: float) -> float:
    return 0.5 * t**2 + c * t + v0

def sDecrease(t: float, c: float, v0: float, s0: float) -> float:
    return -(1.0 / 6.0) * t**3 + 0.5 * c * t**2 + v0 * t + s0

def vDecrease(t: float, c: float, v0: float) -> float:
    return -0.5 * t**2 + c * t + v0

def sMin(t: float, v0: float, s0: float) -> float:
    return -2.0 * t * t + v0 * t + s0

def vMin(t: float, v0: float) -> float:
    return -4.0 * t + v0


# --------------------------------------------
# find_common_tangent
# --------------------------------------------
def find_common_tangent(seg: Segment,
                        a0: float,
                        v0: float,
                        s0: float,
                        s1: float,
                        t0: float,
                        t1: float,
                        k: float) -> Tuple[Point, Point]:
    """
    Перевод C++ find_common_tangent.
    Решает систему:
        velocity1(tA) = velocity2(tB)
        distance1(tA) + velocity1(tA)*(tB - tA) = distance2(tB)

    Метод — итерационный Ньютон.
    """

    # Определяем знак kLast по типу сегмента
    if seg.nameAccel == "I":
        kLast = 1.0
    elif seg.nameAccel == "D":
        kLast = -1.0
    else:
        kLast = 0.0

    def velocity1(tau):
        return velocity(tau, seg.a0, seg.v0, seg.t0, kLast)

    def velocity2(tau):
        return velocity(tau, 0.0, 0.0, t1 - 1.74166, 1.0)

    def distance1(tau):
        return distance(tau, seg.a0, seg.v0, seg.s0, seg.t0, kLast)

    def distance2(tau):
        return distance(tau, 0.0, 0.0, s1 - 3.91389, t1 - 1.74166, 1.0)

    # Начальные приближения
    tA = (seg.t0 + seg.t1) / 2.0
    tB = (t1 + (t1 - 1.74166)) / 2.0

    for _ in range(50):
        F1 = velocity1(tA) - velocity2(tB)
        F2 = distance1(tA) + velocity1(tA) * (tB - tA) - distance2(tB)

        if abs(F1) < 1e-10 and abs(F2) < 1e-10:
            break

        dt = 1e-5

        dF1dtA = (velocity1(tA + dt) - velocity2(tB) - F1) / dt
        dF1dtB = (velocity1(tA) - velocity2(tB + dt) - F1) / dt

        dF2dtA = (
            distance1(tA + dt)
            + velocity1(tA + dt) * (tB - (tA + dt))
            - distance2(tB)
            - F2
        ) / dt

        dF2dtB = (
            distance1(tA)
            + velocity1(tA) * ((tB + dt) - tA)
            - distance2(tB + dt)
            - F2
        ) / dt

        det = dF1dtA * dF2dtB - dF1dtB * dF2dtA
        if abs(det) < 1e-12:
            break

        # Ньютоновское обновление
        tA -= (F1 * dF2dtB - F2 * dF1dtB) / det
        tB -= (dF1dtA * F2 - dF2dtA * F1) / det

    y0 = distance1(tA)
    y1 = distance2(tB)

    point1 = Point(tA, y0)
    point2 = Point(tB, y1)

    return point1, point2

def _shift_keys_up(data: Dict[int, List['Segment']], from_index: int, shift: int = 1) -> None:
    """
    Сдвигает ключи data: для всех ключей >= from_index, ключ := ключ + shift.
    Выполняет модификацию in-place.
    ВАЖНО: предполагаем, что ключи — положительные целые и что нет коллизий при сдвиге.
    """
    if shift <= 0:
        raise ValueError("shift must be positive")
    keys = sorted(k for k in data.keys() if k >= from_index)
    # Проходим в обратном порядке, чтобы не перезаписать
    for k in reversed(keys):
        data[k + shift] = data.pop(k)

def _insert_at_index(data: Dict[int, List['Segment']], index: int, new_vec: List['Segment']) -> None:
    """
    Вставляет новую запись на позицию index, сдвигая все ключи >= index вправо на 1.
    """
    if index in data:
        _shift_keys_up(data, index, shift=1)
    data[index] = new_vec

def findRoot(t0: float,
             a0: float,
             v0: float,
             s0: float,
             nextPoint: Point,
             k: float,
             data: Dict[int, List[Segment]],
             is_diff=False
             ) -> float:

    t1 = nextPoint.t
    s1 = nextPoint.s

    # --- локальные g-функции ---
    def gMax(t, a0_, v0_, s0_, t0_, t1_, s1_):
        return (
            sMax((t - t0_), v0_, s0_)
            + vMax((t - t0_), v0_) * (t1_ - t)
            - (s1_ - safety_distance(t, a0_, v0_, t0_, 1.0))
        )

    def gIncrease(t, a0_, v0_, s0_, t0_, t1_, s1_):
        c = max(2.0, a0_)
        return (
            sIncrease((t - t0_), c, v0_, s0_)
            + vIncrease((t - t0_), c, v0_) * (t1_ - t)
            - (s1_ - safety_distance(t, a0_, v0_, t0_, 1.0))
        )

    def gDecrease(t, a0_, v0_, s0_, t0_, t1_, s1_):
        c = min(-2.0, a0_)
        return (
            sDecrease((t - t0_), c, v0_, s0_)
            + vDecrease((t - t0_), c, v0_) * (t1_ - t)
            - (s1_ - safety_distance(t, a0_, v0_, t0_, -1.0))
        )

    def gMin(t, a0_, v0_, s0_, t0_, t1_, s1_):
        return (
            sMin((t - t0_), v0_, s0_)
            + vMin((t - t0_), v0_) * (t1_ - t)
            - (s1_ - safety_distance(t, a0_, v0_, t0_, -1.0))
        )

    # ==========================================================
    # CASE 1: k == 1.0
    # ==========================================================
    if k == 1.0:

        tAMax = t0 + 4.0 - max(a0, 2.0)

        if tAMax >= t1:

            def g(tau):
                return gIncrease(tau, a0, v0, s0, t0, t1, s1)

            return safe_find_root(g, t0, t1)

        elif a0 == 4.0:

            def g(tau):
                return gMax(tau, a0, v0, s0, t0, t1, s1)

            return safe_find_root(g, t0, t1)

        else:
            def g1(tau):
                return gIncrease(tau, a0, v0, s0, t0, t1, s1)

            root = safe_find_root(g1, t0, tAMax)

            if math.isnan(root):
                aGMax = 4.0
                vGMax = velocity(tAMax, a0, v0, t0, k)
                sGMax = distance(tAMax, a0, v0, s0, t0, k)

                def g2(tau):
                    return gMax(tau, aGMax, vGMax, sGMax, tAMax, t1, s1)

                root = safe_find_root(g2, tAMax, t1)

            return root

    # ==========================================================
    # CASE 2: k == -1.0
    # ==========================================================
    elif k == -1.0:

        tAMax = t0 + 4.0 + min(a0, -2.0)

        if tAMax >= t1:

            def g(tau):
                return gDecrease(tau, a0, v0, s0, t0, t1, s1)

            return safe_find_root(g, t0, t1)

        elif a0 == 4.0:

            def g(tau):
                return gMin(tau, a0, v0, s0, t0, t1, s1)

            return safe_find_root(g, t0, t1)

        else:
            def g1(tau):
                return gDecrease(tau, a0, v0, s0, t0, t1, s1)

            root = safe_find_root(g1, t0, tAMax)

            if math.isnan(root):
                aGMin = -4.0
                vGMin = velocity(tAMax, a0, v0, t0, k)
                sGMin = distance(tAMax, a0, v0, s0, t0, k)

                def g2(tau):
                    return gMin(tau, aGMin, vGMin, sGMin, tAMax, t1, s1)

                root = safe_find_root(g2, tAMax, t1)

            return root
# ----------------------
    # CASE 3: k == 0 -> сложный кейс (копия логики C++)
    # ----------------------
    else:
        if not data:
            return 0.0

        # предполагаем, что ключи целые 1..N (как в C++ коде)
        keys = sorted(data.keys(), reverse=True)
        # PINK TANGENT BLOCK (C++: первый проход)
        for index in keys:
            segments = list(data[index])  # делаем копию списка сегментов
            for j, seg in enumerate(segments):
                if seg.nameAccel in ("I", "D"):
                    kLast = 1.0 if seg.nameAccel == "I" else -1.0
                    fixed_safety_distance = safety_distance(0.0, 0.0, 5.0, 0.0, 1.0)
                    cnt1 = count_intersections(
                        seg.t0, seg.s0, seg.v0,
                        seg.t0, t1 + 100,
                        0.0, 0.0,
                        s1 - 3.91389 - fixed_safety_distance,
                        t1 - 1.74166,
                        1.0
                    )
                    cnt2 = count_intersections(
                        seg.t1, seg.s1, seg.v1,
                        seg.t1, t1 + 100,
                        0.0, 0.0,
                        s1 - 3.91389 - fixed_safety_distance,
                        t1 - 1.74166,
                        1.0
                    )
                    if (cnt1 == 0) != (cnt2 == 0):
                        # найден pink tangent case
                        points = find_common_tangent(seg, a0, v0, s0, s1 - fixed_safety_distance, t0, t1, k)
                        slope = (points[1].s - points[0].s) / (points[1].t - points[0].t)
                        tA = points[0].t
                        if tA < seg.t0:
                            tA = seg.t0
                        tB = points[1].t
                        if tB > t1:
                            # как в C++ — помечаем флаг и break; тут просто пропускаем этот сегмент
                            continue
                        # Вставки/замены в data — воспроизводим ветки way1..way4 из C++.
                        # Для простоты и точности: получаем "size" (как C++ data.size()).
                        max_key = max(data.keys())
                        # Проходим i от index до size (включительно), как в C++
                        i = index
                        while i <= max_key:
                            seg_i = data[i][0]  # первый сег в блоке i
                            seg_i_last = data[i][-1]  # последний сег в блоке i
                            if i == index:
                                if i == max_key:
                                    # way 1 (конец)
                                    if is_diff:
                                        print("way 1")
                                    a1 = acceleration(tA, seg_i.a0, seg_i.t0, kLast)
                                    v1_local = velocity(tA, seg_i.a0, seg_i.v0, seg_i.t0, 1.0)
                                    tEnd = t_for_s_on_line(seg_i.s1, points[0].s, points[0].t, slope)
                                    a2 = acceleration(tB, 0.0, t1 - 1.74166, 1.0)
                                    a3 = acceleration(t1, 0.0, t1 - 1.74166, 1.0)
                                    v2 = velocity(tB, 0.0, 0.0, t1 - 1.74166, 1.0)
                                    v3 = velocity(t1, 0.0, 0.0, t1 - 1.74166, 1.0)
                                    s_safety_1 = safety_distance(tA, seg_i.a0, seg_i.v0, seg_i.t0, kLast)
                                    s_free_1 = seg_i.sTarget - points[0].s - s_safety_1
                                    s_t1_new = distance(seg_i_last.t1, 0.0, slope, points[0].s, tA, 0.0)
                                    s_free_2 = seg_i.sTarget - s_t1_new - s_safety_1
                                    adding = [
                                        Segment(seg_i.t0, tA,
                                                seg_i.nameAccel,
                                                seg_i.a0, a1,
                                                seg_i.v0, v1_local,
                                                seg_i.s0, s_safety_1, points[0].s, s_free_1, seg_i.sTarget),
                                        Segment(tA, seg_i_last.t1,
                                                "C",
                                                0.0, 0.0,
                                                slope, slope,
                                                points[0].s, s_safety_1, s_t1_new, s_free_2, seg_i.sTarget)
                                    ]
                                    # заменить tail (в C++ erase from j to end), в python — делаем так:
                                    old = data[i]
                                    newlist = old[:j] + adding  # сохраним предыдущие сегменты до j (если были)
                                    data[i] = newlist
                                    if is_diff:
                                        display(data, i)
                                    # вставляем следующий блок (i+1) с двумя сегментами
                                    # сдвигаем ключи вправо на 1, чтобы вставить i+1
                                    _shift_keys_up(data, i + 1, shift=1)
                                    data[i + 1] = [
                                        Segment(seg_i_last.t1, tB,
                                                "C",
                                                0.0, 0.0,
                                                slope, slope,
                                                s_t1_new, s_safety_1, points[1].s, (s1 - points[1].s - s_safety_1), s1),
                                        Segment(tB, t1,
                                                "I",
                                                a2, a3,
                                                v2, v3,
                                                points[1].s, fixed_safety_distance, s1 - fixed_safety_distance, 0.0, s1)
                                    ]
                                    if is_diff:
                                        display(data, i + 1)
                                    return 0.0
                                else:
                                    # way 2 (замена внутри, но не конец)
                                    if is_diff:
                                        print("way 2")
                                    a1 = acceleration(tA, seg_i.a0, seg_i.t0, kLast)
                                    v1_local = velocity(tA, seg_i.a0, seg_i.v0, seg_i.t0, 1.0)
                                    s_safety_1 = safety_distance(tA, seg_i.a0, seg_i.v0, seg_i.t0, kLast)
                                    s_free_1 = seg_i.sTarget - points[0].s - s_safety_1
                                    s_t1_new = distance(seg_i_last.t1, 0.0, slope, points[0].s, tA, 0.0)
                                    s_free_2 = seg_i.sTarget - s_t1_new - s_safety_1
                                    adding = [
                                        Segment(seg_i.t0, tA,
                                                seg_i.nameAccel,
                                                seg_i.a0, a1,
                                                seg_i.v0, v1_local,
                                                seg_i.s0, s_safety_1, points[0].s, s_free_1, seg_i.sTarget),
                                        Segment(tA, seg_i_last.t1,
                                                "C",
                                                0.0, 0.0,
                                                slope, slope,
                                                points[0].s, s_safety_1, s_t1_new, s_free_2, seg_i.sTarget)
                                    ]
                                    old = data[i]
                                    newlist = old[:j] + adding
                                    data[i] = newlist
                                    if is_diff:
                                        display(data, i)
                                    # не меняем размер словаря — переходим дальше
                            elif i == max_key:
                                # way 3
                                if is_diff:
                                    print("way 3")
                                prev_seg = data[i - 1][-1]
                                s0_1 = prev_seg.s1
                                s_safety_1 = safety_distance(seg_i_last.t1, 0.0, slope, seg_i.t0, 0.0)
                                s1_1 = distance(seg_i_last.t1, 0.0, slope, s0_1, seg_i.t0, 0.0)
                                s_free_1 = seg_i.sTarget - s1_1 - s_safety_1
                                data[i] = [
                                    Segment(seg_i.t0, seg_i_last.t1,
                                            "C",
                                            0.0, 0.0,
                                            slope, slope,
                                            s0_1, s_safety_1, s1_1, s_free_1, seg_i.sTarget)
                                ]
                                if is_diff:
                                    display(data, i)   
                                a1 = acceleration(tB, 0.0, t1 - 1.74166, 1.0)
                                a2 = acceleration(t1, 0.0, t1 - 1.74166, 1.0)
                                v1_local = velocity(tB, 0.0, 0.0, t1 - 1.74166, 1.0)
                                v2 = velocity(t1, 0.0, 0.0, t1 - 1.74166, 1.0)
                                s_free_2 = s1 - points[1].s - s_safety_1
                                # вставляем i+1
                                _shift_keys_up(data, i + 1, shift=1)
                                data[i + 1] = [
                                    Segment(seg_i_last.t1, tB,
                                            "C",
                                            0.0, 0.0,
                                            slope, slope,
                                            s1_1, s_safety_1, points[1].s, s_free_2, s1),
                                    Segment(tB, t1,
                                            "I",
                                            a1, a2,
                                            v1_local, v2,
                                            points[1].s, fixed_safety_distance, s1 - fixed_safety_distance, 0.0, s1)
                                ]
                                if is_diff:
                                    display(data, i + 1)
                                return 0.0
                            else:
                                # way 4: замена на "C" в середине
                                if is_diff:
                                    print("way 4")
                                prev_seg = data[i - 1][-1]
                                s0_1 = prev_seg.s1
                                s_safety_1 = safety_distance(seg_i_last.t1, 0.0, slope, seg_i.t0, 0.0)
                                s1_1 = distance(seg_i_last.t1, 0.0, slope, s0_1, seg_i.t0, 0.0)
                                s_free_1 = seg_i.sTarget - s1_1 - s_safety_1
                                data[i] = [
                                    Segment(seg_i.t0, seg_i_last.t1,
                                            "C",
                                            0.0, 0.0,
                                            slope, slope,
                                            s0_1, s_safety_1, s1_1, s_free_1, seg_i.sTarget)
                                ]
                                if is_diff:
                                    display(data, i)
                            i += 1
                        return 0.0  # после вставок - вернуть как в C++
        # -------- DOT TANGENT BLOCK (I-only) --------
        flag = False

        if is_diff:
            print("second")

        for index in keys:

            if flag:
                break

            segments = data[index]

            for j in range(len(segments)):

                seg = segments[j]

                if seg.nameAccel == "I":

                    tSwap = -1.0
                    tAMax = seg.t0 + 4.0 - max(seg.a0, 2.0)

                    if tAMax >= t1:

                        def g(tau):
                            return gIncrease(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1)

                        tSwap = safe_find_root(g, seg.t0, t1)

                    elif seg.a0 == 4.0:

                        def g(tau):
                            return gMax(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1)

                        tSwap = safe_find_root(g, seg.t0, t1)

                    else:

                        def g1(tau):
                            return gIncrease(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1)

                        root = safe_find_root(g1, seg.t0, tAMax)

                        if math.isnan(root):

                            aGMax = 4.0
                            vGMax = velocity(tAMax, seg.a0, seg.v0, seg.t0, 1.0)
                            sGMax = distance(tAMax, seg.a0, seg.v0, seg.s0, seg.t0, 1.0)

                            def g2(tau):
                                return gMax(tau, aGMax, vGMax, sGMax, tAMax, t1, s1)

                            root = safe_find_root(g2, tAMax, t1)

                        tSwap = root
                    

                    if math.isnan(tSwap) or tSwap > seg.t1 or tSwap < seg.t0:
                        continue

                    sTSwap = distance(tSwap, seg.a0, seg.v0, seg.s0, seg.t0, 1.0)
                    slope = (s1 - sTSwap) / (t1 - tSwap)

                    if is_diff:
                        print(index, j, slope)

                    # ----------- вычисляем tSwap (точно как в C++) -----------


                    # ----------- ПОЛНОЕ воспроизведение C++ цикла i -----------

                    max_key = max(data.keys())

                    i = index
                    while i <= max_key:
                        
                        seg_i = data[i][0]
                        seg_i_last = data[i][-1]

                        if seg_i.nameAccel == "I" or seg_i.nameAccel == "D":

                            if i == index:
                                # ---------- первый блок ----------
                                a1 = acceleration(tSwap, seg_i.a0, seg_i.t0, 1.0)
                                v1_local = velocity(tSwap, seg_i.a0, seg_i.v0, seg_i.t0, 1.0)
                                s_safety_1 = safety_distance(sTSwap, seg_i.a0, seg_i.v0, seg_i.t0, 1.0)
                                s_free_1 = seg_i.sTarget - sTSwap - s_safety_1

                                s1_2 = distance(seg_i_last.t1, 0.0, slope, sTSwap, tSwap, 0.0)
                                s_free_2 = seg_i.sTarget - s1_2 - s_safety_1

                                adding = [
                                    Segment(
                                        seg_i.t0, tSwap,
                                        seg_i.nameAccel,
                                        seg_i.a0, a1,
                                        seg_i.v0, v1_local,
                                        seg_i.s0, s_safety_1, sTSwap, s_free_1, seg_i.sTarget
                                    ),
                                    Segment(
                                        tSwap, seg_i_last.t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        sTSwap, s_safety_1, s1_2, s_free_2, seg_i.sTarget
                                    )
                                ]

                                old = data[i]
                                data[i] = old[:j] + adding

                                if i == max_key:

                                    # вставляем новый индекс i+1
                                    new_index = max_key + 1
                                    data[new_index] = [
                                        Segment(
                                            seg_i_last.t1, t1,
                                            "C",
                                            0.0, 0.0,
                                            slope, slope,
                                            s1_2, s_safety_1, s1, 0.0, s1
                                        )
                                    ]

                                    flag = True
                                    return 0.0

                            elif i != max_key:

                                # ---------- промежуточный блок ----------
                                s0_1 = data[i - 1][-1].s1
                                s1_1 = distance(seg_i_last.t1, 0.0, slope, s0_1, seg_i.t0, 0.0)
                                s_safety_1 = data[i - 1][-1].safety_s
                                s_free_1 = seg_i.sTarget - s1_1 - s_safety_1

                                data[i] = [
                                    Segment(
                                        seg_i.t0, seg_i_last.t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        s0_1, s_safety_1, s1_1, s_free_1, seg_i.sTarget
                                    )
                                ]

                            else:

                                # ---------- последний блок ----------
                                s0_1 = data[i - 1][-1].s1
                                s1_1 = distance(seg_i_last.t1, 0.0, slope, s0_1, seg_i.t0, 0.0)
                                s_safety_1 = data[i - 1][-1].safety_s
                                s_free_1 = seg_i.sTarget - s1_1 - s_safety_1

                                data[i] = [
                                    Segment(
                                        seg_i.t0, seg_i_last.t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        s0_1, s_safety_1, s1_1, s_free_1, seg_i.sTarget
                                    )
                                ]

                                # вставляем новый индекс i+1
                                new_index = max_key + 1
                                data[new_index] = [
                                    Segment(
                                        seg_i_last.t1, t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        s1_1, s_safety_1, s1, 0.0, s1
                                    )
                                ]

                                flag = True
                                return 0.0

                        i += 1


        for index in keys:

            if flag:
                break

            segments = data[index]

            for j in range(len(segments)):

                seg = segments[j]

                if seg.nameAccel == "I":

                    tSwap = -1.0
                    tAMax = seg.t0 + 4.0 - max(seg.a0, 2.0)

                    if tAMax >= t1:

                        def g(tau):
                            return gIncrease(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1)

                        tSwap = safe_find_root(g, seg.t0, t1)

                    elif seg.a0 == 4.0:

                        def g(tau):
                            return gMax(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1)

                        tSwap = safe_find_root(g, seg.t0, t1)

                    else:

                        def g1(tau):
                            return gIncrease(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1)

                        root = safe_find_root(g1, seg.t0, tAMax)

                        if math.isnan(root):

                            aGMax = 4.0
                            vGMax = velocity(tAMax, seg.a0, seg.v0, seg.t0, 1.0)
                            sGMax = distance(tAMax, seg.a0, seg.v0, seg.s0, seg.t0, 1.0)

                            def g2(tau):
                                return gMax(tau, aGMax, vGMax, sGMax, tAMax, t1, s1)

                            root = safe_find_root(g2, tAMax, t1)

                        tSwap = root
                    

                    if math.isnan(tSwap) or tSwap > seg.t1 or tSwap < seg.t0:
                        continue

                    sTSwap = distance(tSwap, seg.a0, seg.v0, seg.s0, seg.t0, 1.0)
                    slope = (s1 - sTSwap) / (t1 - tSwap)

                    if is_diff:
                        print(index, j, slope)

                    # ----------- вычисляем tSwap (точно как в C++) -----------


                    # ----------- ПОЛНОЕ воспроизведение C++ цикла i -----------

                    max_key = max(data.keys())

                    i = index
                    while i <= max_key:
                        
                        seg_i = data[i][0]
                        seg_i_last = data[i][-1]

                        if seg_i.nameAccel == "I" or seg_i.nameAccel == "D":

                            if i == index:
                                # ---------- первый блок ----------
                                a1 = acceleration(tSwap, seg_i.a0, seg_i.t0, 1.0)
                                v1_local = velocity(tSwap, seg_i.a0, seg_i.v0, seg_i.t0, 1.0)
                                s_safety_1 = safety_distance(sTSwap, seg_i.a0, seg_i.v0, seg_i.t0, 1.0)
                                s_free_1 = seg_i.sTarget - sTSwap - s_safety_1

                                s1_2 = distance(seg_i_last.t1, 0.0, slope, sTSwap, tSwap, 0.0)
                                s_free_2 = seg_i.sTarget - s1_2 - s_safety_1

                                adding = [
                                    Segment(
                                        seg_i.t0, tSwap,
                                        seg_i.nameAccel,
                                        seg_i.a0, a1,
                                        seg_i.v0, v1_local,
                                        seg_i.s0, s_safety_1, sTSwap, s_free_1, seg_i.sTarget
                                    ),
                                    Segment(
                                        tSwap, seg_i_last.t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        sTSwap, s_safety_1, s1_2, s_free_2, seg_i.sTarget
                                    )
                                ]

                                old = data[i]
                                data[i] = old[:j] + adding

                                if i == max_key:

                                    # вставляем новый индекс i+1
                                    new_index = max_key + 1
                                    data[new_index] = [
                                        Segment(
                                            seg_i_last.t1, t1,
                                            "C",
                                            0.0, 0.0,
                                            slope, slope,
                                            s1_2, s_safety_1, s1, 0.0, s1
                                        )
                                    ]

                                    flag = True
                                    return 0.0

                            elif i != max_key:

                                # ---------- промежуточный блок ----------
                                s0_1 = data[i - 1][-1].s1
                                s1_1 = distance(seg_i_last.t1, 0.0, slope, s0_1, seg_i.t0, 0.0)
                                s_safety_1 = data[i - 1][-1].safety_s
                                s_free_1 = seg_i.sTarget - s1_1 - s_safety_1

                                data[i] = [
                                    Segment(
                                        seg_i.t0, seg_i_last.t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        s0_1, s_safety_1, s1_1, s_free_1, seg_i.sTarget
                                    )
                                ]

                            else:

                                # ---------- последний блок ----------
                                s0_1 = data[i - 1][-1].s1
                                s1_1 = distance(seg_i_last.t1, 0.0, slope, s0_1, seg_i.t0, 0.0)
                                s_safety_1 = data[i - 1][-1].safety_s
                                s_free_1 = seg_i.sTarget - s1_1 - s_safety_1

                                data[i] = [
                                    Segment(
                                        seg_i.t0, seg_i_last.t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        s0_1, s_safety_1, s1_1, s_free_1, seg_i.sTarget
                                    )
                                ]

                                # вставляем новый индекс i+1
                                new_index = max_key + 1
                                data[new_index] = [
                                    Segment(
                                        seg_i_last.t1, t1,
                                        "C",
                                        0.0, 0.0,
                                        slope, slope,
                                        s1_1, s_safety_1, s1, 0.0, s1
                                    )
                                ]

                                flag = True
                                return 0.0

                        i += 1



        return 0.0

# ============================================================
# findTimeForDistance
# ============================================================
def findTimeForDistance(S: float,
                        a0: float,
                        v0: float,
                        s0: float,
                        t_left: float,
                        t_right: float,
                        k: float) -> float:

    def F_fixed(t):
        return distance(t, a0, v0, s0, t_left, k) - S

    root = safe_find_root(F_fixed, t_left, t_right)

    if math.isnan(root):
        raise RuntimeError("findTimeForDistance: root not found")

    return root


# ============================================================
# findTimeForVelocity
# ============================================================
def findTimeForVelocity(V: float,
                        a0: float,
                        v0: float,
                        t_left: float,
                        t_right: float,
                        k: float) -> float:

    def F_fixed(t):
        return velocity(t, a0, v0, t_left, k) - V

    root = safe_find_root(F_fixed, t_left, t_right)

    if math.isnan(root):
        raise RuntimeError("findTimeForVelocity: root not found")

    return root


# ============================================================
# display_data
# ============================================================
def display_data(data: Dict[int, List[Segment]]) -> str:

    lines = []
    lines.append("{")

    first_map = True
    for key in data:

        if not first_map:
            lines.append(",")
        first_map = False

        lines.append("    {")
        lines.append(f"        {key},")
        lines.append("        {")

        first_seg = True
        for seg in data[key]:

            if not first_seg:
                lines.append(",")
            first_seg = False

            lines.append("            {")
            lines.append(
                f"                t0 = {seg.t0}, t1 = {seg.t1},"
            )
            lines.append(f'                "{seg.nameAccel}",')
            lines.append(
                f"                a0 = {seg.a0}, a1 = {seg.a1},"
            )
            lines.append(
                f"                v0 = {seg.v0}, v1 = {seg.v1},"
            )
            lines.append(
                f"                s0 = {seg.s0}, safe = {seg.safety_s}, "
                f"s1 = {seg.s1}, s_free = {seg.s_free}, sTarget = {seg.sTarget}"
            )
            lines.append("            }")

        lines.append("        }")
        lines.append("    }")

    lines.append("}")

    return "\n".join(lines)


# ============================================================
# func
# ============================================================
def func(points: Dict[int, Point], sLast, flag=False, flag2 = False):

    if flag:
        print(points)

    data: Dict[int, List[Segment]] = {}

    t0 = 0.0
    a0 = 0.0
    v0 = 0.0
    s0 = 0.0

    a1 = 0.0
    v1 = 0.0

    for i in points:

        t1 = points[i].t
        s1 = points[i].s

        #print(f"S = {s1}    T = {t1}")

        # ====================================================
        # FIRST POINT
        # ====================================================
        if i == 1:

            if (distance(t1, a0, v0, s0, t0, 1.0)
                + safety_distance(t1, a0, v0, t0, 1.0)
                <= s1):

                #print(i, "INCREASE")

                s = distance(t1, a0, v0, s0, t0, 1.0)
                safety_s = safety_distance(t1, a0, v0, t0, 1.0)
                s_free = s1 - s - safety_s

                a1 = acceleration(t1, a0, t0, 1.0)
                v1 = velocity(t1, a0, v0, t0, 1.0)

                data[1] = [
                    Segment(0.0, t1, "I",
                            a0, a1,
                            v0, v1,
                            s0,
                            safety_s,
                            s,
                            s_free,
                            s1)
                ]

                s1 = s
                #print(display_data(data))

            else:

                #print(i, "INCREASE + CONST V")

                nextPoint = Point(t1, s1)
                tSwap = findRoot(t0, a0, v0, s0, nextPoint, 1.0, data)

                a1 = acceleration(tSwap, a0, t0, 1.0)
                v1 = velocity(tSwap, a0, v0, t0, 1.0)

                sTSwap = distance(tSwap, a0, v0, s0, t0, 1.0)
                safety_s = safety_distance(tSwap, a0, v0, t0, 1.0)

                s_free_1 = s1 - safety_s - sTSwap
                s = distance(t1, a1, v1, sTSwap, tSwap, 0.0)
                s_free_2 = s1 - safety_s - s

                data[1] = [
                    Segment(0.0, tSwap, "I",
                            a0, a1,
                            v0, v1,
                            s0,
                            safety_s,
                            sTSwap,
                            s_free_1,
                            s1),

                    Segment(tSwap, t1, "C",
                            0.0, 0.0,
                            v1, v1,
                            sTSwap,
                            safety_s,
                            s,
                            s_free_2,
                            s1)
                ]

                s1 = s
                #print(display_data(data))

            t0 = t1
            a0 = a1
            v0 = v1
            s0 = s1

        # ====================================================
        # OTHER POINTS
        # ====================================================
        else:

            inc_possible = (
                distance(max(t1, t0), a0, v0, s0, t0, 1.0)
                + safety_distance(max(t1, t0), a0, v0, t0, 1.0)
                <= s1
            )

            const_possible = (
                distance(max(t1, t0), a0, v0, s0, t0, 0.0)
                + safety_distance(max(t1, t0), a0, v0, t0, 0.0)
                <= s1
            )

            if flag2:
                dec_possible = (
                    (distance(max(t1, t0), a0, v0, s0, t0, -1.0)
                    + safety_distance(max(t1, t0), a0, v0, t0, -1.0)
                    <= s1) 
                )
            else:
                dec_possible = (
                    (distance(max(t1, t0), a0, v0, s0, t0, -1.0)
                    + safety_distance(max(t1, t0), a0, v0, t0, -1.0)
                    <= s1) and 
                    velocity(max(findRoot(t0, a0, v0, s0, Point(t1, s1), -1.0, data), t0), a0, v0, t0, -1.0) >= 5.0
                )

            if flag:
                print(distance(max(t1, t0), a0, v0, s0, t0, 1.0), safety_distance(max(t1, t0), a0, v0, t0, 1.0), s1)
                print(distance(max(t1, t0), a0, v0, s0, t0, 0.0), safety_distance(max(t1, t0), a0, v0, t0, 0.0), s1)
                print(distance(max(t1, t0), a0, v0, s0, t0, -1.0), safety_distance(max(t1, t0), a0, v0, t0, -1.0), velocity(max(findRoot(t0, a0, v0, s0, Point(t1, s1), -1.0, data), t0), a0, v0, t0, -1.0), s1)

            if inc_possible:
                if flag:
                    print(i, "INCREASE")

                s = distance(t1, a0, v0, s0, t0, 1.0)
                safety_s = safety_distance(t1, a0, v0, t0, 1.0)
                s_free = s1 - s - safety_s

                a1 = acceleration(t1, a0, t0, 1.0)
                v1 = velocity(t1, a0, v0, t0, 1.0)

                data[i] = [
                    Segment(t0, max(t1, t0), "I",
                            a0, a1,
                            v0, v1,
                            s0,
                            safety_s,
                            s,
                            s_free,
                            s1)
                ]

                s1 = s

            elif const_possible:
                if flag:
                    print(i, "INCREASE + CONST V")

                nextPoint = Point(t1, s1)
                tSwap = findRoot(t0, a0, v0, s0, nextPoint, 1.0, data)

                a1 = acceleration(tSwap, a0, t0, 1.0)
                v1 = velocity(tSwap, a0, v0, t0, 1.0)

                sTSwap = distance(tSwap, a0, v0, s0, t0, 1.0)
                safety_s = safety_distance(tSwap, a0, v0, t0, 1.0)

                s_free_1 = s1 - safety_s - sTSwap
                s = distance(t1, a1, v1, sTSwap, tSwap, 0.0)
                s_free_2 = s1 - safety_s - s

                data[i] = [
                    Segment(t0, max(tSwap, t0), "I",
                            a0, a1,
                            v0, v1,
                            s0,
                            safety_s,
                            sTSwap,
                            s_free_1,
                            s1),

                    Segment(max(tSwap, t0), max(t1, t0), "C",
                            0.0, 0.0,
                            v1, v1,
                            sTSwap,
                            safety_s,
                            s,
                            s_free_2,
                            s1)
                ]

                s1 = s

            elif dec_possible:

                if flag:
                    print(i, "DECREASE + CONST V")

                nextPoint = Point(t1, s1)
                tSwap = findRoot(t0, a0, v0, s0, nextPoint, -1.0, data)

                a1 = acceleration(tSwap, a0, t0, -1.0)
                v1 = velocity(tSwap, a0, v0, t0, -1.0)

                sTSwap = distance(tSwap, a0, v0, s0, t0, -1.0)
                safety_s = safety_distance(tSwap, a0, v0, t0, -1.0)

                s_free_1 = s1 - safety_s - sTSwap
                s = distance(t1, a1, v1, sTSwap, tSwap, 0.0)
                s_free_2 = s1 - safety_s - s

                data[i] = [
                    Segment(t0, max(tSwap, t0), "D",
                            a0, a1,
                            v0, v1,
                            s0,
                            safety_s,
                            sTSwap,
                            s_free_1,
                            s1),

                    Segment(max(tSwap, t0), max(t1, t0), "C",
                            0.0, 0.0,
                            v1, v1,
                            sTSwap,
                            safety_s,
                            s,
                            s_free_2,
                            s1)
                ]

                s1 = s

            else:
                nextPoint = Point(t1, s1)

                if flag:
                    print(i, "R")
                
                    findRoot(t0, a0, v0, s0, nextPoint, -6.66, data, True)
                else:
                    findRoot(t0, a0, v0, s0, nextPoint, -6.66, data)


                a1 = data[max(list(data.keys()))][-1].a1
                v1 = data[max(list(data.keys()))][-1].v1
                s1 = data[max(list(data.keys()))][-1].s1

            t0 = max(t1, t0)
            a0 = a1
            v0 = v1
            s0 = s1

        if flag:
            display(data, max(list(data.keys())))
    # #print(display_data(data))
    # #print("=" * 120)

    if flag:
        print(display_data(data))

    last_segment = data[max(list(data.keys()))][-1]
    if last_segment.s1 < sLast:
        tLast = findTimeForDistance(sLast, last_segment.a1, last_segment.v1, last_segment.s1, last_segment.t1, last_segment.t1 + 20, 1.0)

        a1 = acceleration(tLast, last_segment.a1, last_segment.t1, 1.0)
        v1 = velocity(tLast, last_segment.a1, last_segment.v1, last_segment.t1, 1.0)
        safety_s = safety_distance(tLast, last_segment.a1, last_segment.v1, last_segment.t1, 1.0)
        s1 = last_segment.s1 + safety_s
        data[max(list(data.keys())) + 1] = [
                    Segment(last_segment.t1, tLast, "I",
                            last_segment.a1, a1,
                            last_segment.v1, v1,
                            last_segment.s1,
                            safety_s,
                            sLast,
                            0.0,
                            s1
                            )
                ]

    return data

