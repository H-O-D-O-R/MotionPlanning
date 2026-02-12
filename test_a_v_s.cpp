#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include <functional>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <boost/math/quadrature/gauss_kronrod.hpp>
using boost::math::quadrature::gauss_kronrod;
using boost::math::tools::eps_tolerance;
using boost::math::tools::toms748_solve;
using namespace std;

struct Segment {
    double t0;
    double t1;
    string nameAccel;
    double a0;
    double a1;
    double v0;
    double v1;
    double s0;
    double s1;
};

struct Point {
    double t;
    double s;
};



void display(map<int, vector<Segment>> m, int i) {
    for (const auto& seg : m[i]) {
        cout << "  Interval: ["
            << seg.t0 << ", "
            << seg.t1 << "] - "
            << seg.nameAccel << ", "
            << "a0 = " << seg.a0 << ", "
            << "a1 = " << seg.a1 << ", "
            << "v0 = " << seg.v0 << ", "
            << "v1 = " << seg.v1 << ", "
            << "s0 = " << seg.s0 << ", "
            << "s1 = " << seg.s1 << "\n";
    }
}



double acceleration(double t1, double a0, double t0, double k) {
    if (k > 0) {
        return min(max(2.0, a0) + k * (t1 - t0), 4.0);
    }
    return max(min(-2.0, a0) + k * (t1 - t0), -4.0);
}

double velocity(double t1, double a0, double v0, double t0, double k)
{
    auto aIncrease = [](double t, double a0, double t0) {
        return min(max(2.0, a0) + 1.0 * (t - t0), 4.0);
        };

    auto aDecrease = [](double t, double a0, double t0) {
        return min(-2.0, a0) - 1.0 * (t - t0);
        };

    if (k == 1.0) {
        auto tAMax = t0 + 4.0 - max(a0, 2.0);
        if ((tAMax - t0) >= (t1 - t0)) {
            auto a = [&](double tau) {
                return aIncrease(tau, a0, t0);
                };
            double integral = gauss_kronrod<double, 61>::integrate(a, t0, t1);

            return v0 + integral;
        }
        else if (a0 == 4.0) {
            double integral = 4 * (t1 - t0);

            return v0 + integral;
        }
        else {
            auto a1 = [&](double tau) {
                return aIncrease(tau, a0, t0);
                };
            double integral1 = gauss_kronrod<double, 61>::integrate(a1, t0, tAMax);

            double integral2 = 4 * (t1 - tAMax);

            return v0 + integral1 + integral2;
        }
    }
    else if (k == 0.0) {
        return v0;
    }
    else {
        auto tAMax = t0 + 4.0 + min(a0, -2.0);
        if ((tAMax - t0) >= (t1 - t0)) {
            auto a = [&](double tau) {
                return aDecrease(tau, a0, t0);
                };
            double integral = gauss_kronrod<double, 61>::integrate(a, t0, t1);

            return v0 + integral;
        }
        else if (a0 == 4.0) {
            double integral = 4 * (t1 - t0);

            return v0 + integral;
        }
        else {
            auto a1 = [&](double tau) {
                return aDecrease(tau, a0, t0);
                };
            double integral1 = gauss_kronrod<double, 61>::integrate(a1, t0, tAMax);

            double integral2 = 4 * (t1 - tAMax);

            return v0 + integral1 + integral2;
        }
    }


}

double distance(double t1, double a0, double v0, double s0, double t0, double k)
{
    auto v = [&](double tau) {
        return velocity(tau, a0, v0, t0, k);
        };
    double integral = gauss_kronrod<double, 61>::integrate(v, t0, t1);

    return s0 + integral;
}



int countIntersections(double t2, double s2, double slope, double t_min, double t_max, double a0, double v0, double s0, double t0, double k)
{
    auto F = [&](double t) {
        double s_curve = distance(t, a0, v0, s0, t0, k);
        double s_line = s2 + slope * (t - t2);
        return s_curve - s_line;
        };

    int count = 0;

    const int N = 200;        // ������������� (����� ���������)
    double dt = (t_max - t_min) / N;

    double prev_t = t_min;
    double prev_F = F(prev_t);

    for (int i = 1; i <= N; i++)
    {
        double t = t_min + i * dt;
        double val = F(t);

        // �������� ����� �����
        if (prev_F == 0.0)
        {
            count++;
        }
        else if (prev_F * val < 0.0)
        {
            // ����� �������� � ������ � ��������
            eps_tolerance<double> tol(50);
            boost::uintmax_t max_iter = 50;

            auto r = toms748_solve(F, prev_t, t, tol, max_iter);
            double root = (r.first + r.second) / 2.0;

            if (root > t2) // ������� ������
                count++;
        }

        prev_t = t;
        prev_F = val;
    }

    return count;
}

int isAboveLine(double t0, double s0, double slope, double t, double s) {
    double s_line = s0 + slope * (t - t0);
    if (s > s_line) {
        return 1; // ����� ��� ������
    }
    else {
        return 0; // ����� �� ����� ��� ��� ������
    }
}




double t_for_s_on_line(double s_target, double s2, double t2, double slope)
{
    // slope �� ������ ���� ����, ����� ����� ��������������
    if (std::abs(slope) < 1e-12)
        throw std::runtime_error("Slope is zero � cannot solve for t.");

    return t2 + (s_target - s2) / slope;
}

double t_for_s_on_curve(double s_target, double t_min, double t_max, double a0, double v0, double s0, double t0, double k)
{
    auto F = [&](double t) {
        return distance(t, a0, v0, s0, t0, k) - s_target;
        };

    // �������� �����
    double f_min = F(t_min);
    double f_max = F(t_max);

    if (f_min * f_max > 0)
        throw std::runtime_error("t_for_s_on_curve: root is not bracketed!");

    boost::math::tools::eps_tolerance<double> tol(50);
    boost::uintmax_t max_iter = 50;

    auto result = boost::math::tools::toms748_solve(
        F, t_min, t_max,
        tol,  // ���������� ��������
        max_iter
    );

    // ���������� �������� ���������
    return (result.first + result.second) * 0.5;
}




double safeFindRoot(auto g, double t0, double t1) {
    try {
        boost::math::tools::eps_tolerance<double> tol(50);
        boost::uintmax_t max_iter = 50;

        auto result = boost::math::tools::toms748_solve(
            g,
            t0,
            t1,
            tol,
            max_iter
        );

        double root = (result.first + result.second) / 2.0;
        return root;
    }
    catch (const std::exception& e) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double sMax(double t, double v0, double s0) {
    return 2 * t * t + v0 * t + s0;
};
double vMax(double t, double v0) {
    return 4 * t + v0;
};
double sIncrease(double t, double c, double v0, double s0) {
    return 1.0 / 6.0 * t * t * t + 0.5 * c * t * t + v0 * t + s0;
};
double vIncrease(double t, double c, double v0) {
    return 0.5 * t * t + c * t + v0;
};
double sDecrease(double t, double c, double v0, double s0) {
    return -1.0 / 6.0 * t * t * t + 0.5 * c * t * t + v0 * t + s0;
};
double vDecrease(double t, double c, double v0) {
    return -0.5 * t * t + c * t + v0;
};
double sMin(double t, double v0, double s0) {
    return -2 * t * t + v0 * t + s0;
};
double vMin(double t, double v0) {
    return -4 * t + v0;
};

pair<Point, Point> find_common_tangent(Segment seg, double a0, double v0, double s0, double s1, double t0, double t1, double k) {

    double kLast = 0.0;
    if (seg.nameAccel == "I") {
        kLast = 1.0;
    }
    else if (seg.nameAccel == "D") {
        kLast = -1.0;
    }

    auto velocity1 = [&](double tau) {
        return velocity(tau, seg.a0, seg.v0, seg.t0, kLast);
        };
    auto velocity2 = [&](double tau) {
        return velocity(tau, 0.0, 0.0, t1 - 1.74166, 1.0);
        };
    auto distance1 = [&](double tau) {
        return distance(tau, seg.a0, seg.v0, seg.s0, seg.t0, kLast);
        };
    auto distance2 = [&](double tau) {
        return distance(tau, 0.0, 0.0, s1 - 3.91389, t1 - 1.74166, 1.0);
        };


    double tA_guess = (seg.t0 + seg.t1) / 2;
    double tB_guess = (t1 + t1 - 1.74166) / 2;

    // ��������� �����������
    double x[2] = { tA_guess, tB_guess };

    // ���������� �����: ����������� ����������� (����� �������� �� ����� ������ solver)
    for (int iter = 0; iter < 50; ++iter) {
        double tA = x[0], tB = x[1];
        double F1 = velocity1(tA) - velocity2(tB);
        double F2 = distance1(tA) + velocity1(tA) * (tB - tA) - distance2(tB);

        if (abs(F1) < 1e-10 && abs(F2) < 1e-10) break;

        // ���������� ��������� (�������� ��� ������ �������)
        double dt = 1e-5;
        double dF1dtA = (velocity1(tA + dt) - velocity2(tB) - F1) / dt;
        double dF1dtB = (velocity1(tA) - velocity2(tB + dt) - F1) / dt;
        double dF2dtA = (distance1(tA + dt) + velocity1(tA + dt) * (tB - (tA + dt)) - distance2(tB) - F2) / dt;
        double dF2dtB = (distance1(tA) + velocity1(tA) * (tB + dt - tA) - distance2(tB + dt) - F2) / dt;

        double det = dF1dtA * dF2dtB - dF1dtB * dF2dtA;
        if (abs(det) < 1e-12) break;

        // ����������
        x[0] -= (F1 * dF2dtB - F2 * dF1dtB) / det;
        x[1] -= (dF1dtA * F2 - dF2dtA * F1) / det;
    }

    double y0 = distance1(x[0]);
    double y1 = distance2(x[1]);

    Point point1 = { x[0], y0 };
    Point point2 = { x[1], y1 };

    return { point1, point2 };
}

double findRoot(double t0, double a0, double v0, double s0, Point nextPoint, double k, map<int, vector<Segment>> data = {}) {

    double t1 = nextPoint.t;
    double s1 = nextPoint.s;

    auto gMax = [](double t, double a0, double v0, double s0, double t0, double t1, double s1) {
        return sMax((t - t0), v0, s0) + vMax((t - t0), v0) * (t1 - t) - s1;
        };
    auto gIncrease = [](double t, double a0, double v0, double s0, double t0, double t1, double s1) {
        auto c = max(2.0, a0);
        return sIncrease((t - t0), c, v0, s0) + vIncrease((t - t0), c, v0) * (t1 - t) - s1;
        };
    auto gDecrease = [](double t, double a0, double v0, double s0, double t0, double t1, double s1) {
        auto c = min(-2.0, a0);
        return sDecrease((t - t0), c, v0, s0) + vDecrease((t - t0), c, v0) * (t1 - t) - s1;
        };
    auto gMin = [](double t, double a0, double v0, double s0, double t0, double t1, double s1) {
        auto c = min(-2.0, a0);
        return sMin((t - t0), v0, s0) + vMin((t - t0), v0) * (t1 - t) - s1;
        };

    if (k == 1.0) {
        auto tAMax = t0 + 4.0 - max(a0, 2.0);
        if (tAMax >= t1) {
            auto g = [&](double tau) {
                return gIncrease(tau, a0, v0, s0, t0, t1, s1);
                };

            eps_tolerance<double> tol(50);
            boost::uintmax_t max_iter = 50;

            auto result = toms748_solve(
                g,
                t0,
                t1,
                tol,
                max_iter
            );

            double root = (result.first + result.second) / 2.0;

            return root;

        }
        else if (a0 == 4.0) {
            auto g = [&](double tau) {
                return gMax(tau, a0, v0, s0, t0, t1, s1);
                };

            eps_tolerance<double> tol(50);
            boost::uintmax_t max_iter = 50;

            auto result = toms748_solve(
                g,
                t0,
                t1,
                tol,
                max_iter
            );

            double root = (result.first + result.second) / 2.0;

            return root;
        }
        else {
            auto g1 = [&](double tau) {
                return gIncrease(tau, a0, v0, s0, t0, t1, s1);
                };

            auto root = safeFindRoot(g1, t0, tAMax);
            if (isnan(root)) {
                auto aGMax = 4.0;
                auto vGMax = velocity(tAMax, a0, v0, t0, k);
                auto sGMax = distance(tAMax, a0, v0, s0, t0, k);

                auto g2 = [&](double tau) {
                    return gMax(tau, aGMax, vGMax, sGMax, tAMax, t1, s1);
                    };

                root = safeFindRoot(g2, tAMax, t1);
            }
            return root;
        }
    }
    else if (k == -1.0) {
        auto tAMax = t0 + 4.0 + min(a0, -2.0);
        if (tAMax >= t1) {
            auto g = [&](double tau) {
                return gDecrease(tau, a0, v0, s0, t0, t1, s1);
                };

            eps_tolerance<double> tol(50);
            boost::uintmax_t max_iter = 50;

            auto result = toms748_solve(
                g,
                t0,
                t1,
                tol,
                max_iter
            );

            double root = (result.first + result.second) / 2.0;

            return root;

        }
        else if (a0 == 4.0) {
            auto g = [&](double tau) {
                return gMin(tau, a0, v0, s0, t0, t1, s1);
                };

            eps_tolerance<double> tol(50);
            boost::uintmax_t max_iter = 50;

            auto result = toms748_solve(
                g,
                t0,
                t1,
                tol,
                max_iter
            );

            double root = (result.first + result.second) / 2.0;

            return root;
        }
        else {
            auto g1 = [&](double tau) {
                return gDecrease(tau, a0, v0, s0, t0, t1, s1);
                };

            auto root = safeFindRoot(g1, t0, tAMax);
            if (root == NAN) {
                auto aGMin = -4.0;
                auto vGMin = velocity(tAMax, a0, v0, t0, k);
                auto sGMin = distance(tAMax, a0, v0, s0, t0, k);

                auto g2 = [&](double tau) {
                    return gMax(tau, aGMin, vGMin, sGMin, tAMax, t1, s1);
                    };

                root = safeFindRoot(g2, tAMax, t1);
            }
            return root;
        }
    }
    else {
        //��������� ������ ������ ��������� �����, ����� v � ����� {t1, s1} ���� ����� 5
        //����� ����������� � ����� �������
        double tA = -1.0;
        double tB = -1.0;

        vector<int> keys;
        for (auto& p : data) keys.push_back(p.first);
        reverse(keys.begin(), keys.end());

        int flag = 0;
        for (int index : keys) {
            if (flag) {
                break;
            }
            cout << "index = " << index << endl;
            display(data, index);
            auto segments = data[index];

            for (int j = 0; j < segments.size(); j++) {
                cout << "j = " << j << endl;
                auto seg = segments[j];

                if ((seg.nameAccel == "I") or (seg.nameAccel == "D")) {
                    cout << "j = " << j << " I or D" << endl;

                    double kLast = 0.0;
                    if (seg.nameAccel == "I") {
                        kLast = 1.0;
                    }
                    else if (seg.nameAccel == "D") {
                        kLast = -1.0;
                    }

                    auto cnt1 = countIntersections(seg.t0, seg.s0, seg.v0, seg.t0, t1 + 100, 0.0, 0.0, s1 - 3.91389, t1 - 1.74166, 1.0);
                    auto cnt2 = countIntersections(seg.t1, seg.s1, seg.v1, seg.t1, t1 + 100, 0.0, 0.0, s1 - 3.91389, t1 - 1.74166, 1.0);

                    cout << "cnt1 = " << cnt1 << " cnt2 = " << cnt2 << endl;

                    if ((cnt1 == 0) != (cnt2 == 0)) {
                        cout << "j = " << j << " has tangent for pink" << endl;

                        auto points = find_common_tangent(seg, a0, v0, s0, s1, t0, t1, k);
                        double slope = (points.second.s - points.first.s) / (points.second.t - points.first.t);

                        tA = points.first.t;
                        tB = points.second.t;

                        //�������� ����������� � ������ ������� ��� ���
                        if (tB > t1) {
                            cout << "j = " << j << " hasn't tangent for pink" << endl;
                            flag = 1;
                            break;
                        }

                        //�������� ��� ������ ��������, ������� �����

                        for (int i = index; i <= data.size(); i++) {
                            cout << "i = " << i << endl;
                            auto seg = data[i][0];
                            if (i == index) {
                                if (i == data.size()) {
                                    if (points.second.s >= seg.s1) {
                                        cout << "way 1" << endl;
                                        auto a1 = acceleration(tA, seg.a0, seg.t0, kLast);
                                        auto v1 = velocity(tA, seg.a0, seg.v0, seg.t0, 1.0);

                                        auto tEnd = t_for_s_on_line(seg.s1, points.first.s, points.first.t, slope);

                                        auto a2 = acceleration(tB, 0.0, t1 - 1.74166, 1.0);
                                        auto a3 = acceleration(t1, 0.0, t1 - 1.74166, 1.0);

                                        auto v2 = velocity(tB, 0.0, 0.0, t1 - 1.74166, 1.0);
                                        auto v3 = velocity(t1, 0.0, 0.0, t1 - 1.74166, 1.0);



                                        vector<Segment> adding = {
                                            { seg.t0, tA, seg.nameAccel, seg.a0, a1, seg.v0, v1, seg.s0, points.first.s},
                                            { tA, tEnd, "C", 0.0, 0.0, slope, slope, points.first.s, seg.s1}
                                        };

                                        data[i].erase(data[i].begin() + j, data[i].end());
                                        data[i].insert(data[i].end(), adding.begin(), adding.end());

                                        display(data, i);


                                        data[i + 1] = {
                                            { tEnd, tB, "C", 0.0, 0.0, slope, slope, seg.s1, points.second.s},
                                            { tB, t1, "I", a2, a3, v2, v3, points.second.s, s1}
                                        };

                                        display(data, i + 1);

                                        break;

                                    }
                                    else {
                                        cout << "way 2" << endl;
                                        auto a1 = acceleration(tA, seg.a0, seg.t0, kLast);
                                        auto v1 = velocity(tA, seg.a0, seg.v0, seg.t0, 1.0);

                                        auto tEnd = t_for_s_on_curve(seg.s1, t1 - 1.74166, t1, 0.0, 0.0, s1 - 3.91389, t1 - 1.74166, 1.0);

                                        auto a2 = acceleration(tB, 0.0, t1 - 1.74166, 1.0);
                                        auto a3 = acceleration(tEnd, 0.0, t1 - 1.74166, 1.0);
                                        auto a4 = acceleration(t1, 0.0, t1 - 1.74166, 1.0);

                                        auto v2 = velocity(tB, 0.0, 0.0, t1 - 1.74166, 1.0);
                                        auto v3 = velocity(tEnd, 0.0, 0.0, t1 - 1.74166, 1.0);
                                        auto v4 = velocity(t1, 0.0, 0.0, t1 - 1.74166, 1.0);

                                        vector<Segment> adding = {
                                            { seg.t0, tA, seg.nameAccel, seg.a0, a1, seg.v0, v1, seg.s0, points.first.s},
                                            { tA, tB, "C", 0.0, 0.0, slope, slope, points.first.s, points.second.s},
                                            { tB, tEnd, "I", a2, a3, v2, v3, points.second.s, seg.s1}
                                        };

                                        data[i].erase(data[i].begin() + j, data[i].end());
                                        data[i].insert(data[i].end(), adding.begin(), adding.end());

                                        display(data, i);

                                        data[i + 1] = {
                                            { tEnd, t1, "I", a3, a4, v3, v4, seg.s1, s1}
                                        };

                                        display(data, i + 1);

                                        break;

                                    }
                                }
                                else {
                                    cout << "way 3" << endl;
                                    auto a1 = acceleration(tA, seg.a0, seg.t0, kLast);
                                    auto v1 = velocity(tA, seg.a0, seg.v0, seg.t0, 1.0);


                                    auto tEnd = t_for_s_on_line(seg.s1, points.first.s, points.first.t, slope);

                                    auto sTA = distance(tA, seg.a0, seg.v0, seg.s0, seg.t0, kLast);

                                    vector<Segment> adding = {
                                        { seg.t0, tA, seg.nameAccel, seg.a0, a1, seg.v0, v1, seg.s0, sTA},
                                        { tA, tEnd, "C", 0.0, 0.0, slope, slope, sTA, seg.s1}
                                    };


                                    data[i].erase(data[i].begin() + j, data[i].end());
                                    data[i].insert(data[i].end(), adding.begin(), adding.end());

                                    display(data, i);
                                }

                            }
                            else if (i == data.size()) {
                                if (points.second.s >= seg.s1) {
                                    cout << "way 4" << endl;
                                    auto tStart = t_for_s_on_line(seg.s0, points.first.s, points.first.t, slope);
                                    auto tEnd = t_for_s_on_line(seg.s1, points.first.s, points.first.t, slope);

                                    data[i] = {
                                       { tStart, tEnd, "C", 0.0, 0.0, slope, slope, seg.s0, seg.s1}
                                    };


                                    display(data, i);

                                    auto a1 = acceleration(tB, 0.0, t1 - 1.74166, 1.0);
                                    auto a2 = acceleration(t1, 0.0, t1 - 1.74166, 1.0);

                                    auto v1 = velocity(tB, 0.0, 0.0, t1 - 1.74166, 1.0);
                                    auto v2 = velocity(t1, 0.0, 0.0, t1 - 1.74166, 1.0);

                                    data[i + 1] = {
                                        { tEnd, tB, "C", 0.0, 0.0, slope, slope, seg.s1, points.second.s},
                                        { tB, t1, "I", a1, a2, v1, v2, points.second.s, s1}
                                    };

                                    display(data, i + 1);

                                    break;


                                }
                                else {
                                    cout << "way 5" << endl;
                                    auto tStart = t_for_s_on_line(seg.s0, points.first.s, points.first.t, slope);
                                    auto tEnd = t_for_s_on_curve(seg.s1, t1 - 1.74166, t1, 0.0, 0.0, s1 - 3.91389, t1 - 1.74166, 1.0);

                                    auto a1 = acceleration(tB, 0.0, t1 - 1.74166, 1.0);
                                    auto a2 = acceleration(tEnd, 0.0, t1 - 1.74166, 1.0);
                                    auto a3 = acceleration(t1, 0.0, t1 - 1.74166, 1.0);

                                    auto v1 = velocity(tB, 0.0, 0.0, t1 - 1.74166, 1.0);
                                    auto v2 = velocity(tEnd, 0.0, 0.0, t1 - 1.74166, 1.0);
                                    auto v3 = velocity(t1, 0.0, 0.0, t1 - 1.74166, 1.0);

                                    data[i] = {
                                        { tStart, tB, "C", 0.0, 0.0, slope, slope, seg.s0, points.second.s},
                                        { tB, tEnd, "I", a1, a2, v1, v2, points.second.s, seg.s1}
                                    };

                                    display(data, i);


                                    data[i + 1] = {
                                        { tEnd, t1, "I", a2, a3, v2, v3, seg.s1, s1}
                                    };

                                    display(data, i + 1);

                                    break;
                                }
                            }
                            else {
                                cout << "way 6" << endl;
                                auto tStart = t_for_s_on_line(seg.s0, points.first.s, points.first.t, slope);
                                auto tEnd = t_for_s_on_line(seg.s1, points.first.s, points.first.t, slope);

                                data[i] = {
                                   { tStart, tEnd, "C", 0.0, 0.0, slope, slope, seg.s0, seg.s1}
                                };

                                display(data, i);

                            }
                        }

                        return 0.0;
                    }
                }
            }
        }

        //����� ������ � ����� ����������� � I � �����
        flag = 0;
        for (int index : keys) {
            if (flag) {
                break;
            }
            cout << "index = " << index << endl;
            display(data, index);
            auto segments = data[index];

            for (int j = 0; j < segments.size(); j++) {
                cout << "j = " << j << endl;
                auto seg = segments[j];

                if ((seg.nameAccel == "I")) {
                    cout << "j = " << j << " I" << endl;

                    auto cnt1 = isAboveLine(seg.t0, seg.s0, seg.v0, t1, s1);
                    auto cnt2 = isAboveLine(seg.t1, seg.s1, seg.v1, t1, s1);

                    if (cnt1 != cnt2) {
                        cout << "j = " << j << " has tangent for dot" << endl;

                        auto tSwap = -1.0;
                        auto tAMax = seg.t0 + 4.0 - max(seg.a0, 2.0);
                        if (tAMax >= t1) {
                            auto g = [&](double tau) {
                                return gIncrease(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1);
                                };

                            eps_tolerance<double> tol(50);
                            boost::uintmax_t max_iter = 50;

                            auto result = toms748_solve(
                                g,
                                seg.t0,
                                t1,
                                tol,
                                max_iter
                            );

                            double root = (result.first + result.second) / 2.0;

                            tSwap = root;

                        }
                        else if (seg.a0 == 4.0) {
                            auto g = [&](double tau) {
                                return gMax(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1);
                                };

                            eps_tolerance<double> tol(50);
                            boost::uintmax_t max_iter = 50;

                            auto result = toms748_solve(
                                g,
                                seg.t0,
                                t1,
                                tol,
                                max_iter
                            );

                            double root = (result.first + result.second) / 2.0;

                            tSwap = root;
                        }
                        else {
                            auto g1 = [&](double tau) {
                                return gIncrease(tau, seg.a0, seg.v0, seg.s0, seg.t0, t1, s1);
                                };

                            auto root = safeFindRoot(g1, seg.t0, tAMax);
                            if (isnan(root)) {
                                auto aGMax = 4.0;
                                auto vGMax = velocity(tAMax, seg.a0, seg.v0, seg.t0, 1.0);
                                auto sGMax = distance(tAMax, seg.a0, seg.v0, seg.s0, seg.t0, 1.0);

                                auto g2 = [&](double tau) {
                                    return gMax(tau, aGMax, vGMax, sGMax, tAMax, t1, s1);
                                    };

                                root = safeFindRoot(g2, tAMax, t1);
                            }

                            tSwap = root;
                        }

                        auto sTSwap = distance(tSwap, seg.a0, seg.v0, seg.s0, seg.t0, 1.0);
                        auto slope = (s1 - sTSwap) / (t1 - tSwap);


                        //�������� ��� ������ ��������, ������� �����

                        for (int i = index; i <= data.size(); i++) {
                            cout << "i = " << i << endl;
                            auto seg = data[i][0];
                            if (i == index) {

                                auto tEnd = t_for_s_on_line(seg.s1, sTSwap, tSwap, slope);

                                auto a1 = acceleration(tSwap, seg.a0, seg.t0, 1.0);
                                auto v1 = velocity(tSwap, seg.a0, seg.v0, seg.t0, 1.0);

                                vector<Segment> adding = {
                                    { seg.t0, tSwap, seg.nameAccel, seg.a0, a1, seg.v0, v1, seg.s0, sTSwap},
                                    { tSwap, tEnd, "C", 0.0, 0.0, slope, slope, sTSwap, seg.s1}
                                };

                                data[i].erase(data[i].begin() + j, data[i].end());
                                data[i].insert(data[i].end(), adding.begin(), adding.end());

                                display(data, i);
                            }
                            else if (i != data.size()) {
                                auto tStart = t_for_s_on_line(seg.s0, sTSwap, tSwap, slope);
                                auto tEnd = t_for_s_on_line(seg.s1, sTSwap, tSwap, slope);

                                data[i] = {
                                       { tStart, tEnd, "C", 0.0, 0.0, slope, slope, seg.s0, seg.s1}
                                };
                                display(data, i);
                            }
                            else {
                                auto tStart = t_for_s_on_line(seg.s0, sTSwap, tSwap, slope);
                                auto tEnd = t_for_s_on_line(seg.s1, sTSwap, tSwap, slope);

                                data[i] = {
                                       { tStart, tEnd, "C", 0.0, 0.0, slope, slope, seg.s0, seg.s1}
                                };
                                display(data, i);

                                data[i + 1] = {
                                       { tEnd, t1, "C", 0.0, 0.0, slope, slope, seg.s1, s1}
                                };
                                display(data, i + 1);
                                flag = 1;
                                return 0.0;
                            }
                        }
                    }
                }
            }
        }
        return 0.0;
    }
}



double findTimeForDistance(double S, double a0, double v0, double s0, double t_left, double t_right, double k)
{
    auto F_fixed = [&](double t) {
        return distance(t, a0, v0, s0, t_left, k) - S;
        };

    eps_tolerance<double> tol(50);
    boost::uintmax_t max_iter = 50;

    auto result = toms748_solve(
        F_fixed,
        t_left,
        t_right,
        tol,
        max_iter
    );

    return (result.first + result.second) / 2.0;
}

double findTimeForVelocity(double V, double a0, double v0, double t_left, double t_right, double k)
{
    auto F_fixed = [&](double t) {
        return velocity(t_right, a0, v0, t_left, k) - V;
        };

    eps_tolerance<double> tol(50);
    boost::uintmax_t max_iter = 50;

    auto result = toms748_solve(
        F_fixed,
        t_left,
        t_right,
        tol,
        max_iter
    );

    return (result.first + result.second) / 2.0;
}


std::string display_data(const std::map<int, std::vector<Segment>>& data) {
    std::ostringstream oss;

    oss << "{\n";

    bool first_map = true;
    for (const auto& [key, vec] : data) {
        if (!first_map) oss << ",\n";
        first_map = false;

        oss << "    {\n";
        oss << "        " << key << ",\n";
        oss << "        {\n";

        bool first_seg = true;
        for (const auto& seg : vec) {
            if (!first_seg) oss << ",\n";
            first_seg = false;

            oss << "            {\n";
            oss << "                "
                << seg.t0 << ", " << seg.t1 << ",\n";
            oss << "                \"" << seg.nameAccel << "\",\n";
            oss << "                " << seg.a0 << ", " << seg.a1 << ",\n";
            oss << "                " << seg.v0 << ", " << seg.v1 << ",\n";
            oss << "                " << seg.s0 << ", " << seg.s1 << "\n";
            oss << "            }";
        }

        oss << "\n        }\n";
        oss << "    }";
    }

    oss << "\n}";

    return oss.str();
}


string func(map<int, Point> points) {
    map<int, vector<Segment>> data;


    map<int, vector<Segment>> a;
    map<int, vector<Segment>> v;



    auto t0 = 0.0;
    auto a0 = 0.0;
    auto v0 = 0.0;
    auto s0 = 0.0;

    auto a1 = 0.0;
    auto v1 = 0.0;

    for (const auto& point : points) {
        auto i = point.first;

        auto t1 = point.second.t;
        auto s1 = point.second.s;
        cout << "S = " << s1 << "    T = " << t1 << endl;

        if (i == 1) {
            if (distance(t1, a0, v0, s0, t0, 1.0) <= s1) {
                cout << i << " INCREASE" << endl;

                t1 = findTimeForDistance(s1, a0, v0, s0, t0, t0 + 10, 1.0);

                a1 = acceleration(t1, a0, t0, 1.0);
                v1 = velocity(t1, a0, v0, t0, 1.0);

                data[1] = { { 0.0, t1, "I", a0, a1, v0, v1, s0, s1} };

                display(data, 1);
            }
            else {
                cout << i << " INCREASE + CONST V" << endl;

                Point nextPoint = { t1, s1 };
                auto tSwap = findRoot(t0, a0, v0, s0, nextPoint, 1.0);


                a1 = acceleration(tSwap, a0, t0, 1.0);
                v1 = velocity(tSwap, a0, v0, t0, 1.0);

                auto sTSwap = distance(tSwap, a0, v0, s0, t0, 1.0);

                data[1] = {
                    { 0.0, tSwap, "I", a0, a1, v0, v1, s0, sTSwap },
                    { tSwap, t1, "C", 0, 0, v1, v1, sTSwap, s1}
                };

                display(data, 1);
            };

            t0 = t1;
            a0 = a1;
            v0 = v1;
            s0 = s1;

        }
        else {
            if (distance(t1, a0, v0, s0, t0, 1.0) <= s1) {
                cout << i << " INCREASE" << endl;

                t1 = findTimeForDistance(s1, a0, v0, s0, t0, t0 + 10, 1.0);

                a1 = acceleration(t1, a0, t0, 1.0);
                v1 = velocity(t1, a0, v0, t0, 1.0);

                data[i] = {
                    { t0, t1, "I", a0, a1, v0, v1, s0, s1 }
                };

                display(data, i);
            }
            else if (distance(t1, a0, v0, s0, t0, 0.0) <= s1) {
                cout << i << " INCREASE + CONST V" << endl;

                Point nextPoint = { t1, s1 };
                auto tSwap = findRoot(t0, a0, v0, s0, nextPoint, 1.0);

                a1 = acceleration(tSwap, a0, t0, 1.0);
                v1 = velocity(tSwap, a0, v0, t0, 1.0);

                auto sTSwap = distance(tSwap, a0, v0, s0, t0, 1.0);

                data[i] = {
                    { t0, tSwap, "I", a0, a1, v0, v1, s0, sTSwap },
                    { tSwap, t1, "C", 0, 0, v1, v1, sTSwap, s1}
                };

                display(data, i);
            }
            else if ((distance(t1, a0, v0, s0, t0, -1.0) <= s1) and (velocity(t1, a0, v0, t0, -1.0) >= 5.0)) {
                cout << i << " DECREASE + CONST V" << endl;

                Point nextPoint = { t1, s1 };
                auto tSwap = findRoot(t0, a0, v0, s0, nextPoint, -1.0);
                a1 = acceleration(tSwap, a0, t0, -1.0);
                v1 = velocity(tSwap, a0, v0, t0, -1.0);

                auto sTSwap = distance(tSwap, a0, v0, s0, t0, -1.0);

                data[i] = {
                    { t0, tSwap, "D", a0, a1, v0, v1, s0, sTSwap },
                    { tSwap, t1, "C", 0, 0, v1, v1, sTSwap, s1}
                };

                display(data, i);
            }
            else {
                cout << i << " R" << endl;

                Point nextPoint = { t1, s1 };
                auto tSwap = findRoot(t0, a0, v0, s0, nextPoint, -6.66, data);
            }

            t0 = t1;
            a0 = a1;
            v0 = v1;
            s0 = s1;
        }
    };
    
    cout << display_data(data) << endl;
    cout << "==========================================================================================================" << endl << endl << endl;
    return display_data(data);
}


