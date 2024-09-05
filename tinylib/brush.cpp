#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstring>

struct Point2D
{
    double x = 0;
    double y = 0;
};

Point2D sum(const Point2D& p1, const Point2D& p2) { return Point2D{p1.x+p2.x, p1.y+p2.y}; }
Point2D diff(const Point2D& p1, const Point2D& p2) { return Point2D{p1.x-p2.x, p1.y-p2.y}; }
Point2D mul(const Point2D& p, double f) { return Point2D{p.x*f, p.y*f}; }
Point2D mid(const Point2D& p1, const Point2D& p2) { return mul(sum(p1, p2), 0.5); }
double norm(const Point2D& v) { return sqrt(v.x*v.x + v.y*v.y); }
double manhattanLength(const Point2D& v) { return fabs(v.x) + fabs(v.y); }
double distance(const Point2D& p1, const Point2D& p2) { return norm(diff(p1, p2)); }

struct Line2D
{
    Point2D p1;
    Point2D p2;
};

double length(const Line2D& l) { return distance(l.p1, l.p2); }

//return a point which is p2 moved in the direction of the line s.t. the length
//of the line from p1 to this new point is newLength
Point2D changeLength(const Line2D& l, double newLength)
{
    if(l.p1.x==l.p2.x && l.p1.y==l.p2.y) { //length is 0 - "line" has no direction
        //to move p2 in
        return l.p2;
    }
    Point2D d = diff(l.p2, l.p1);
    double length = norm(d);
    return sum(l.p1, mul(d, newLength/length));
}

double lineToPointDistance(const Line2D& l, const Point2D& p)
{
    Point2D v = diff(l.p2, l.p1);
    Point2D w = diff(p, l.p1);
    double cross = v.x * w.y - v.y * w.x;
    return fabs(cross) / norm(v);
}

enum class Intersection
{
    NONE,
    BOUNDED,
    UNBOUNDED
};

Intersection intersection(const Line2D& l1, const Line2D& l2, Point2D *intersectionPoint)
{
    Point2D a = diff(l1.p2, l1.p1);
    Point2D b = diff(l2.p1, l2.p2);
    Point2D c = diff(l1.p1, l2.p1);

    double denominator = a.y * b.x - a.x * b.y;
    if(denominator == 0 || !std::isfinite(denominator)) {
        return Intersection::NONE;
    }

    double reciprocal = 1 / denominator;
    double na = (b.y * c.x - b.x * c.y) * reciprocal;
    if(intersectionPoint) {
        *intersectionPoint = sum(l1.p1, mul(a, na));
    }

    if(na < 0 || na > 1) {
        return Intersection::UNBOUNDED;
    }

    double nb = (a.x * c.y - a.y * c.x) * reciprocal;
    if(nb < 0 || nb > 1) {
        return Intersection::UNBOUNDED;
    }

    return Intersection::BOUNDED;
}

class ImagePainter
{
  public:
    //output pixel data
    unsigned char* _image = nullptr;
    int _width = 0;
    int _height = 0;
    int _xstride = 0;
    int _ystride = 0;
    bool _erase = false;

    void drawLine(const Point2D& start, const Point2D& end, double width);
};

//TODO: LUT
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

void ImagePainter::drawLine(const Point2D& start, const Point2D& end, double width)
{
    double lineMag = distance(start, end);
    double invSqLineMag = 1/(lineMag*lineMag);
    Point2D d = diff(end, start);

    auto distFromLine = [&](const Point2D& p) {
        if(lineMag < 1e-8) { //the line segment is a point
            return distance(p, start);
        }
        //project the point onto the line segment
        Point2D pd = diff(p, start);
        double u = ( pd.x*d.x + pd.y*d.y ) * invSqLineMag;
        //clip to [0,1]
        u = std::min(1.0, std::max(0.0, u));

        //projection point
        Point2D proj = sum(start, mul(d, u));

        //return the distance from the projection point
        return distance(p, proj);
    };
    int startx = floor(std::min(start.x,end.x) - width);
    int starty = floor(std::min(start.y,end.y) - width);
    int endx = ceil(std::max(start.x,end.x) + width);
    int endy = ceil(std::max(start.y,end.y) + width);
    double halfWidth = width * 0.5;

    double w = 2; //empirically, things get grainy below w=2 (say at 1.5), and at w=2,
    //the narrowest line width where we're guaranteed to have 255s all along it
    //for a 4-connected flood fill to be stopped by the line is 2.5

    for(int y = starty; y < endy; y++) {
        if(y < 0 || y >= _height) {
            continue;
        }
        for(int x = startx; x < endx; x++) {
            if(x < 0 || x >= _width) {
                continue;
            }
            double dist = distFromLine({(double)x,(double)y});

            if(dist > halfWidth+w+1) {
                continue;
            }

            int grey = 255;
            double c = std::max(-w, std::min(w, dist - halfWidth));
            grey = (1-sigmoid(c*6/w)) * 255;
            if(grey >= 255 - 30) {
                grey = 255;
            }
            int ind = y*_ystride + x*_xstride;
            if(_erase) {
                _image[ind] = std::min(255-grey, (int)_image[ind]);
            }
            else {
                _image[ind] = std::max(grey, (int)_image[ind]);
            }
        }
    }
}

struct SamplePoint
{
    Point2D pos;
    double time = 0;
    double pressure = 0;
};

Point2D tangent(const SamplePoint& p1, const SamplePoint& p2)
{
    Point2D d = diff(p1.pos, p2.pos);
    double scale = 1 / std::max(1.0, p1.time - p2.time);
    return Point2D{d.x*scale, d.y*scale};
}

SamplePoint mix(const Point2D& p, double t, const SamplePoint& p1, const SamplePoint& p2)
{
    SamplePoint s;
    s.pos = p;
    s.pressure = (1 - t) * p2.pressure + t * p1.pressure;
    s.time = (1 - t) * p2.time + t * p1.time;
    return s;
}

enum class Smoothing
{
    NONE,
    SIMPLE,
    WEIGHTED,
};

class Brush
{
  public:
    //config params
    Smoothing _smoothing = Smoothing::NONE;
    double _smoothDist = 0;
    double _tailAggressiveness = 0;
    bool _smoothPressure = false;
    double _lineWidth = 3;

    //output image
    ImagePainter* _painter = nullptr;

    void initPaint(const SamplePoint& p);
    //smoothing modifies p
    void paint(SamplePoint& p, double zoomCoeff);
    void endPaint();

    void paintAt(const SamplePoint& p);
    void paintLine(const SamplePoint& p1, const SamplePoint& p2);
    void paintBezierSegment(const SamplePoint& p1, const SamplePoint& p2, const Point2D& tangent1, const Point2D& tangent2);
    void paintBezierCurve(const SamplePoint& p1, const SamplePoint& p2, const Point2D& control1, const Point2D& control2);
  private:
    //painting state
    SamplePoint _prevP; //previous value of p passed to paint (or initPaint the first time paint is called)
    SamplePoint _olderP; //previous value of _prevP
    std::vector<SamplePoint> _history;
    std::vector<double> _distHist;
    bool _paintedAtLeastOnce = false;
    bool _haveTangent = false;
    Point2D _prevTangent{0, 0};
};

void Brush::initPaint(const SamplePoint& p)
{
    _paintedAtLeastOnce = false;
    _haveTangent = false;
    _prevTangent = Point2D{0, 0};
    _prevP = p;
    _history.clear();
    _distHist.clear();
}

void Brush::paint(SamplePoint& p, double zoomCoeff)
{
    if(_smoothing == Smoothing::WEIGHTED && _smoothDist > 0) {
        //smooth the coordinates by taking a weighted average of the history positions & pressure
        Point2D prevPos = _history.empty() ? _prevP.pos : _history.back().pos;
        _distHist.push_back(distance(p.pos, prevPos));

        _history.push_back(p);

        double x=0, y=0;

        if(_history.size() > 3) {
            //we always use "smoothing distance"; Krita has this as a parameter
            double sigma = _smoothDist * zoomCoeff / 3.0;

            double gaussianWeight = 1 / (sqrt(2 * M_PI) * sigma);
            double gaussianWeight2 = sigma * sigma;
            double distanceSum = 0;
            double scaleSum = 0;
            double pressure = 0;
            double baseRate = 0;

            assert(_history.size() == _distHist.size());

            for(int i=_history.size()-1; i>=0; --i) {
                double rate = 0;

                const SamplePoint& nextP = _history[i];
                double distance = _distHist[i];
                assert(distance >= 0);

                double pressureGrad = 0;
                if(i < (int)_history.size()-1) {
                    pressureGrad = nextP.pressure - _history[i+1].pressure;

                    if(pressureGrad > 0) {
                        pressureGrad *= _tailAggressiveness * 40 * (1 - nextP.pressure);
                        distance += pressureGrad * 3 * sigma;
                    }
                }

                if(gaussianWeight2 != 0) {
                    distanceSum += distance;
                    rate = gaussianWeight * exp(-distanceSum * distanceSum / (2 * gaussianWeight2));
                }

                if(_history.size()-i == 1) {
                    baseRate = rate;
                }
                else if(baseRate / rate > 100) {
                    break;
                }

                scaleSum += rate;
                x += rate * nextP.pos.x;
                y += rate * nextP.pos.y;
                if(_smoothPressure) {
                    pressure += rate * nextP.pressure;
                }
            }

            if(scaleSum != 0) {
                x /= scaleSum;
                y /= scaleSum;
                if(_smoothPressure) {
                    pressure /= scaleSum;
                }
            }

            if((x != 0 && y != 0) || (x == p.pos.x && y == p.pos.y)) {
                p.pos.x = x;
                p.pos.y = y;
                if(_smoothPressure) {
                    p.pressure = pressure;
                }
                _history.back() = p;
            }
        }
    }

    if(_smoothing == Smoothing::SIMPLE || _smoothing == Smoothing::WEIGHTED) {
        if(!_haveTangent) {
            _haveTangent = true;
            _prevTangent = tangent(p, _prevP);
        }
        else {
            Point2D newTangent = tangent(p, _olderP);
            if((newTangent.x == 0 && newTangent.y == 0) || (_prevTangent.x == 0 && _prevTangent.y == 0)) {
                paintLine(_prevP, p);
            }
            else {
                paintBezierSegment(_olderP, _prevP, _prevTangent, newTangent);
            }
            _prevTangent = newTangent;
        }
        _olderP = _prevP;
    }
    else if(_smoothing == Smoothing::NONE) {
        paintLine(_prevP, p);
    }
    _prevP = p;
}

void Brush::endPaint()
{
    if(!_paintedAtLeastOnce) {
        paintAt(_prevP);
    }
    else if(_smoothing != Smoothing::NONE && _haveTangent) {
        _haveTangent = false;
        Point2D newTangent = tangent(_prevP, _olderP);
        paintBezierSegment(_olderP, _prevP, _prevTangent, newTangent);
    }
}

void Brush::paintBezierSegment(const SamplePoint& p1, const SamplePoint& p2, const Point2D& tangent1, const Point2D& tangent2)
{
    if(tangent1.x == 0 && tangent1.y == 0) {
        return;
    }
    if(tangent2.x == 0 && tangent2.y == 0) {
        return;
    }

    double maxSanePoint = 1e6;

    Point2D controlTarget1;
    Point2D controlTarget2;

    Point2D controlDirection1 = sum(p1.pos, tangent1);
    Point2D controlDirection2 = diff(p2.pos, tangent2);

    Line2D line1{p1.pos, controlDirection1};
    Line2D line2{p2.pos, controlDirection2};

    Line2D line3{controlDirection1, controlDirection2};
    Line2D line4{p1.pos, p2.pos};

    Point2D inter;
    if(intersection(line3, line4, &inter) == Intersection::BOUNDED) {
        double controlLength = length(line4) / 2;
        controlTarget1 = changeLength(line1, controlLength);
        controlTarget2 = changeLength(line2, controlLength);
    }
    else {
        Intersection type = intersection(line1, line2, &inter);
        if(type == Intersection::NONE || manhattanLength(inter) > maxSanePoint) {
            inter = mul(sum(p1.pos, p2.pos), 0.5);
        }
        controlTarget1 = inter;
        controlTarget2 = inter;
    }

    double coeff = 0.8;

    double velocity1 = norm(tangent1);
    double velocity2 = norm(tangent2);

    if(velocity1 == 0 || velocity2 == 0) {
        velocity1 = 1e-6;
        velocity2 = 1e-6;
        //not a case we should actually reach...
    }

    double similarity = std::min(velocity1/velocity2, velocity2/velocity1);
    similarity = std::max(similarity, 0.5);
    coeff *= 1 - std::max(0.0, similarity - 0.8);

    assert(coeff > 0);

    Point2D control1;
    Point2D control2;

    if(velocity1 > velocity2) {
        control1 = sum(mul(p1.pos, (1.0 - coeff)), mul(controlTarget1, coeff));
        coeff *= similarity;
        control2 = sum(mul(p2.pos, (1.0 - coeff)), mul(controlTarget2, coeff));
    }
    else {
        control2 = sum(mul(p2.pos, (1.0 - coeff)), mul(controlTarget2, coeff));
        coeff *= similarity;
        control1 = sum(mul(p1.pos, (1.0 - coeff)), mul(controlTarget1, coeff));
    }

    paintBezierCurve(p1, p2, control1, control2);
}

const double BEZIER_FLATNESS_THRESHOLD = 0.3; //Krita uses 0.5; smaller is smoother and potentially more expensive

void Brush::paintBezierCurve(const SamplePoint& p1, const SamplePoint& p2, const Point2D& control1, const Point2D& control2)
{
    Line2D line{p1.pos, p2.pos};
    double d1 = lineToPointDistance(line, control1);
    double d2 = lineToPointDistance(line, control2);

    if((d1 < BEZIER_FLATNESS_THRESHOLD && d2 < BEZIER_FLATNESS_THRESHOLD) || std::isnan(d1) || std::isnan(d2)) {
        paintLine(p1, p2);
    }
    else {
        Point2D l2 = mid(p1.pos, control1);
        Point2D h = mid(control1, control2);
        Point2D l3 = mid(l2, h);
        Point2D r3 = mid(control2, p2.pos);
        Point2D r2 = mid(h, r3);
        Point2D l4 = mid(l3, r2);

        SamplePoint midP = mix(l4, 0.5, p1, p2);

        paintBezierCurve(p1, midP, l2, l3);
        paintBezierCurve(midP, p2, r2, r3);
    }
}

void Brush::paintLine(const SamplePoint& p1, const SamplePoint& p2)
{
    _paintedAtLeastOnce = true;
    if(_painter) {
        _painter->drawLine(p1.pos, p2.pos, _lineWidth);
    }
    else {
      printf("{%f,%f},\n", p1.pos.x, p1.pos.y);
      printf("{%f,%f},\n", p2.pos.x, p2.pos.y);
    }
}

void Brush::paintAt(const SamplePoint& p)
{
    SamplePoint peps = p;
    peps.pos.x += 0.001;
    paintLine(p, peps);
}

//extern "C" API

extern "C" Brush* brush_init_paint(double x, double y, double time, double lineWidth, double smoothDist, int erase, unsigned char* image, int width, int height, int xstride, int ystride)
{
    Brush& brush = *new Brush;
    brush._smoothing = Smoothing::WEIGHTED;
    brush._lineWidth = lineWidth;
    brush._smoothDist = smoothDist;
    brush._tailAggressiveness = 0; //since we don't have pressure values, this parameter has no effect anyway
    
    ImagePainter& painter = *new ImagePainter;
    painter._image = image;
    painter._width = width;
    painter._height = height;
    painter._xstride = xstride;
    painter._ystride = ystride;
    painter._erase = erase;
    brush._painter = &painter;

    brush.initPaint({{x,y},time});

    return &brush;
}

//TODO: return affected ROI
extern "C" void brush_paint(Brush* brush, double* x, double* y, double time, double zoom)
{
    SamplePoint s{{*x,*y},time};
    brush->paint(s, zoom);
    *x = s.pos.x;
    *y = s.pos.y;
}

//TODO: return affected ROI
extern "C" void brush_end_paint(Brush* brush)
{
    brush->endPaint();
    delete brush->_painter;
    delete brush;
}

