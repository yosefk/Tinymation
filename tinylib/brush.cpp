#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

struct Point2D
{
    double x = 0;
    double y = 0;
    bool operator==(const Point2D& p) const { return x==p.x && y==p.y; }
    bool operator!=(const Point2D& p) const { return !(*this == p); }
};

//you don't really want to use it, but for quick hacks, it's there
namespace std {
    template <>
    struct hash<Point2D> {
        size_t operator()(const Point2D& p) const noexcept {
            // Combine hashes of x and y (example using std::hash)
            size_t h1 = std::hash<int>{}(p.x);
            size_t h2 = std::hash<int>{}(p.y);
            return h1 ^ (h2 << 1); // Simple hash combination
        }
    };
}

Point2D sum(const Point2D& p1, const Point2D& p2) { return Point2D{p1.x+p2.x, p1.y+p2.y}; }
Point2D diff(const Point2D& p1, const Point2D& p2) { return Point2D{p1.x-p2.x, p1.y-p2.y}; }
Point2D mul(const Point2D& p, double f) { return Point2D{p.x*f, p.y*f}; }
Point2D mid(const Point2D& p1, const Point2D& p2) { return mul(sum(p1, p2), 0.5); }
double dot(const Point2D& p1, const Point2D& p2) { return p1.x*p2.x + p1.y*p2.y; }
double norm(const Point2D& v) { return sqrt(v.x*v.x + v.y*v.y); }
double manhattanLength(const Point2D& v) { return fabs(v.x) + fabs(v.y); }
double distance(const Point2D& p1, const Point2D& p2) { return norm(diff(p1, p2)); }


double cosOfAngleBetween2Lines(const Point2D& p1, const Point2D& p2, const Point2D& p3) {
    // Compute direction vectors: v1 = p2 - p1, v2 = p3 - p2
    Point2D v1 = diff(p2, p1); // Vector from p1 to p2
    Point2D v2 = diff(p3, p2); // Vector from p2 to p3

    // Compute magnitudes of the vectors
    double mag_v1 = norm(v1);
    double mag_v2 = norm(v2);

    double dot_product = dot(v1, v2);

    // Compute cosine of the angle
    double cos_theta = dot_product / (mag_v1 * mag_v2);

    // Clamp cos_theta to [-1, 1] to handle numerical errors
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;

    return cos_theta;
}

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

auto projectPointOntoLineSegument(const Point2D& start, const Point2D& end)
{
    double lineMag = distance(start, end);
    double invSqLineMag = 1/(lineMag*lineMag);
    Point2D d = diff(end, start);

    return [=](const Point2D& p, double& raw_u) {
        if(lineMag < 1e-8) { //the line segment is a point
            return start;
        }
        //project the point onto the line segment
        Point2D pd = diff(p, start);
        double u = ( pd.x*d.x + pd.y*d.y ) * invSqLineMag;
        raw_u = u;
        //clip to [0,1]
        u = std::min(1.0, std::max(0.0, u));

        //projection point
        Point2D proj = sum(start, mul(d, u));

        return proj;
    };
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

struct SamplePoint
{
    Point2D pos;
    double time = 0;
    double pressure = 0;
    bool operator==(const SamplePoint& p) const {
        return pos == p.pos && time == p.time && pressure == p.pressure;
    }
    bool operator!=(const SamplePoint& p) const { return !(*this == p); }
};

unsigned int g_random_seed = 1;

class Noise2D
{
  private:
    //TODO: better seeding
    std::mt19937 rng{g_random_seed++};         // Random number generator
    std::uniform_real_distribution<double> angleDist{0.0, 2.0 * M_PI}; // For random angle (0 to 2π)
    std::uniform_real_distribution<double> magDist{0, 1};   // For random magnitude (0 to 1)

  public:
    // Helper function to add random noise to a point
    Point2D addNoise(const Point2D& point, double maxNoise) {
        if (maxNoise <= 0) {
            return point;
        }
        double angle = angleDist(rng);
        //sqrt would produce the same distribution all over the circle instead of higher density closer to the center
        //as we'd get if we didn't correct for a circle closer to the center being shorter
        //we use pow 0.7 to have a degree of pull towards the center (lowest-pressure pencil which has the highest
        //circle center noise looks good with this setting, specifically)
        double magnitude = pow(magDist(rng), 0.7) * maxNoise;
        return Point2D{
            point.x + magnitude * cos(angle),
            point.y + magnitude * sin(angle)
        };
    }
};

struct Coord
{
    int x=0;
    int y=0;
    Coord neigh(int xoft, int yoft) const { return {x+xoft,y+yoft}; }
};

class CoordSet
{
  public:
    struct Value
    {
        unsigned char pixVal=0;
        bool valid=false;
    };
    void setRegion(int x, int y, int width, int height) {
        if(_values.size() < width*height) {
            _values.resize(width*height);
        }
        _x = x;
        _y = y;
        _width = width;
        _height = height;
        std::fill(_values.begin(), _values.end(), Value());
    }
    void setRegion(const Point2D& center, int radius) {
        int x = (int)floor(center.x - radius);
        int y = (int)floor(center.y - radius);
        int width = (int)ceil(center.x + radius) - x;
        int height = (int)ceil(center.y + radius) - y;
        setRegion(x,y,width,height);
    }
    void addPixel(int x, int y, int pixVal) {
        if(outOfBounds(x,y)) {
            return;
        }
        Value& v = _values[index(x,y)];
        v.valid = true;
        v.pixVal = pixVal;
    }
    Value findPixel(int x, int y) const {
        if(outOfBounds(x,y)) {
            return Value();
        }
        return _values[index(x,y)];
    }

  private:
    bool outOfBounds(int x, int y) const {
        return x < _x || x >= _x + _width || y < _y || y >= _y + _height;
    }
    int index(int x, int y) const { return (y-_y)*_width + (x-_x); }
    int _x = 0;
    int _y = 0;
    int _width = 0;
    int _height = 0;
    std::vector<Value> _values;
};

//the "pix traits" business is for being able to call drawSoftCircle on both alpha channels
//(for line rendering/erasing) and RGB (where we paint the color and leave alpha alone.)
//note drawing lines with non-noisy soft circles gives a good line but not as well anti-aliased
//as drawLine() produces; but this should be good enough for _line coloring_ where you see few edges
//between lines of different colors, and this is much easier to implement than do the various odd
//things drawLine does in the RGB space.

struct PixTraitsAlpha
{
    typedef unsigned char PixVal;
    PixVal immutable(bool erase) const { return erase ? 0 : 255; }
    PixVal fetch(const unsigned char* pixAddr) const { return *pixAddr; }
    void store(unsigned char* pixAddr, PixVal pixVal) const { *pixAddr = pixVal; }
    PixVal blend(unsigned char pixValChange, PixVal oldVal) const {
        int newVal = (pixValChange*255 + (255-pixValChange)*oldVal + (1<<7)) >> 8;
        newVal = std::max(newVal, (int)oldVal); //otherwise newVal can get smaller by 1 which adds up badly
        if(newVal >= 255-10) {
            newVal = 255;
        }
        return newVal;
    }
    PixVal erase(double pressure, double intensity, PixVal oldVal) const {
        return std::max(0, std::min(int(oldVal * ((1-pressure) * intensity) + oldVal * (1-intensity)), 255));
    }
    int toInt(PixVal val) const { return val; }
};

struct RGBPixVal
{
    unsigned char rgb[3];
    bool operator==(const RGBPixVal& that) const { return rgb[0]==that.rgb[0] && rgb[1]==that.rgb[1] && rgb[2]==that.rgb[2]; }
    bool operator!=(const RGBPixVal& that) const { return !(*this == that); } 
};

struct PixTraitsRGB
{
    typedef RGBPixVal PixVal;
    PixVal newColor;

    PixVal immutable(bool erase) const { return newColor; }
    PixVal fetch(const unsigned char* pixAddr) const {
        PixVal v;
        v.rgb[0] = pixAddr[0];
        v.rgb[1] = pixAddr[1];
        v.rgb[2] = pixAddr[2];
        return v;
    }
    void store(unsigned char* pixAddr, PixVal v) const {
        pixAddr[0] = v.rgb[0];
        pixAddr[1] = v.rgb[1];
        pixAddr[2] = v.rgb[2];
    }
    PixVal blend(unsigned char pixValChange, PixVal oldVal) const {
        PixVal v;
        if(pixValChange == 0) {
            return oldVal;
        }
        for(int i=0; i<3; ++i) {
            v.rgb[i] = (pixValChange*newColor.rgb[i] + (255-pixValChange)*oldVal.rgb[i] + (1<<7)) >> 8;
        }
        return v;
    }
    PixVal erase(double pressure, double intensity, PixVal oldVal) const {
        return PixVal(); //not implemented/used ATM
    }
    int toInt(const PixVal&) const { return 0; }
};

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

    int _xmin;
    int _ymin;
    int _xmax;
    int _ymax;

    bool _rgb = false;
    PixTraitsRGB _pixTraitsRGB;

    ImagePainter() { resetROI(); }

    virtual ~ImagePainter() {}
    virtual void onPixelPainted(int x, int y, int value) {}
    virtual void onLinePainted(const Point2D& start, const Point2D& end) {}

    void resetROI();
    void getROI(int* region);
    void paintWithin(const int* region);
    //this is a "pen" line - solid with sharp antialised edges
    void drawLine(const Point2D& start, const Point2D& end, double width, const unsigned char* rgb=nullptr);

    //this is inspired by Krita's "Pencil-2"; it draws circles with the diameter equaling the line width,
    //placing them not too close to each other (it remembers the last center across calls)
    //and adding some noise to the center location 
    void drawLineUsingWideSoftCiclesWithNoisyCenters(const SamplePoint& start, const SamplePoint& end, double width);

    void detectSharpTurns(bool b) { _detectSharpTurns=b; if(!b) _cumDist=0; }

  private:
    int value(const Coord& c) { return _image[_ystride*c.y + _xstride*c.x]; }
    //currently this function can do two kinds of changes to the image pixels:
    //- if greatestPixValChange is positive, it does an additive change (it computes an intensity
    //  per pixel to make the circle edges soft, and then adds greatestPixValChange * intensity, clamping to 255)
    //- otherwise it does a multiplicative change using the pressure parameter, more accurately it interpolates
    //  between the old value and a new one computed by multiplying by 1-pressure, with intensity defining the weights
    //  (high intensity / middle of the circle -> more change.)
    template<class PixTraits>
    void drawSoftCircle(const Point2D& center, double radius, int greatestPixValChange, double pressure, const PixTraits& pixTraits);

    //these are used by drawLineUsingWideSoftCiclesWithNoisyCenters
    Point2D _lastCircleCenter; // Center of the last drawn circle
    double _remainingDistance=0; // Distance until the next circle should be drawn
    bool _isFirstSegment=true;      // True for the first segment of a polyline

    Noise2D _noise2D;

    ///don't paint outside this rectangle
    Point2D _minPainted;
    Point2D _maxPainted;

    //drawLine maintains the set of coordinates around the previous end coordinate
    //(passed to the previous drawLine call) to avoid artifacts of alpha blending
    CoordSet _aroundPrevEndpoint;
    CoordSet _aroundCurrEndpoint;

    //drawLine detects "sharp turns" in the polyline since otherwise its artifact
    //avoidance for the points where polyline segments connect creates its own artifacts
    //upon sharp turns
    Point2D _lastStart; //we need the previous start coordinate to detect a turn
    bool _lastStartValid = false;
    //we turn off sharp turn detection at the end (maybe too late but hopefully it prevents
    //at least some "fat dots" when raising the pen), and more shockingly at the beginning
    //(because tablets produce strange coordinates early on sometimes and this paints ugly
    //as it is without the additional line fatness when a sharp turn is detected in those
    //weird coordinates.) so we ignore sharp turns until we accumulate enough polyline distance
    //from the start (which means a sharp turn right at the beginning can show the artifact
    //sharp turn detection is supposed to prevent... not too bad for a cleanup pen though
    //and the "real" solution is to somehow deal with the bad input we sometimes get)
    bool _detectSharpTurns = false;
    double _cumDist = 0;
};

//TODO: LUT
double sigmoid(double x) { return 1 / (1 + exp(-x)); }

template<class PixTraits>
void ImagePainter::drawSoftCircle(const Point2D& center, double radius, int greatestPixValChange, double pressure, const PixTraits& pix)
{
    // Extend the bounding box slightly to capture anti-aliased edges
    double aa_margin = 1.0; // Extra pixels for smooth edges
    int startx = floor(std::max(center.x - radius - aa_margin, _minPainted.x));
    int starty = floor(std::max(center.y - radius - aa_margin, _minPainted.y));
    int endx = ceil(std::min(center.x + radius + aa_margin, _maxPainted.x));
    int endy = ceil(std::min(center.y + radius + aa_margin, _maxPainted.y));

    typename PixTraits::PixVal immutableVal = pix.immutable(_erase);
    for(int y = starty; y < endy; y++) {
        for(int x = startx; x < endx; x++) {
            int ind = y*_ystride + x*_xstride;
            typename PixTraits::PixVal oldVal = pix.fetch(_image+ind);
            if(oldVal == immutableVal) {
                continue;
            }
            double d = distance(center, Point2D{double(x+0.5),double(y+0.5)});

            // Anti-aliasing: Smooth transition near the edge
            const double aa_width = 1.0; // Width of the anti-aliasing band
            const double feather_width = 3;
            double intensity = 0.0;

            if(radius < feather_width) {
                if (d <= radius - aa_width) {
                    // Fully inside the circle: use linear falloff
                    intensity = 1.0 - d / radius;
                } else if (d < radius + aa_width) {
                    // In the anti-aliasing region: blend linear falloff with smooth edge
                    double t = (d - radius) / aa_width; // t in [-1, 1]
                    double s = 0.5 - 0.5 * t; // Linearly maps t from [-1,1] to [0,1]
                    double aa_factor = s * s * (3.0 - 2.0 * s); // Smoothstep for edge
                    // Linearly interpolate intensity from (1 - d/radius) at inner edge
                    intensity = (1.0 - d / radius) * aa_factor;
                } else {
                    // Outside the circle
                    continue;
                }
            }
            else {
                if (d <= radius) {// - aa_width) {
                    double min_falloff_dist = radius - feather_width;
                    double capped_distance = std::max(d, min_falloff_dist);
                    intensity = (1.0 - (capped_distance - min_falloff_dist) / feather_width);
                } else {
                    // Outside the circle
                    continue;
                }
            }
            intensity = std::max(intensity, 0.0);

            // Apply pixel value change
            int pixValChange = round(double(greatestPixValChange) * intensity);
            typename PixTraits::PixVal newVal;
            if(greatestPixValChange > 0) {
                //newVal = std::max(0, std::min(oldVal + pixValChange, 255));
                //newVal = (pixValChange*255 + (255-pixValChange)*oldVal + (1<<7)) >> 8;
                //newVal = std::max(newVal, oldVal); //otherwise newVal can get smaller by 1 which adds up badly
                //if(newVal >= 255-10) {
                //    newVal = 255;
                //}
                newVal = pix.blend(pixValChange, oldVal);
            }
            else {
                //newVal = std::max(0, std::min(int(oldVal * ((1-pressure) * intensity) + oldVal * (1-intensity)), 255));
                newVal = pix.erase(pressure, intensity, oldVal);
            }

            if(newVal != oldVal) {
                pix.store(_image+ind, newVal);
                _xmin = std::min(_xmin, x);
                _ymin = std::min(_ymin, y);
                _xmax = std::max(_xmax, x);
                _ymax = std::max(_ymax, y);
                if(!_rgb) {
                    onPixelPainted(x, y, pix.toInt(newVal));
                }
            }
        }
    }
}

void ImagePainter::drawLineUsingWideSoftCiclesWithNoisyCenters(const SamplePoint& starts, const SamplePoint& ends, double width)
{
    Point2D start = starts.pos;
    Point2D end = ends.pos;

    double radius = width / 2;
    double interval = _erase ? radius * 0.3 : radius * 0.1;
    double pressure = starts.pressure;// + ends.pressure)/2; //TODO: move this into the loop instead of being loop invariant
    int greatestPixValChange = 0;

    double maxCenterNoise = radius * 0.25;

    if(_rgb) {
        maxCenterNoise = 0;
        greatestPixValChange = 64;
    }

    auto updatePressureParams = [&] {
        if(!_erase && !_rgb) {
            greatestPixValChange = 2 + 96*std::min(1., std::max(0., pressure*pressure - 0.1 + pressure*0.2)) * (_erase ? -1 : 1);//std::max(0., pressure*pressure - 0.1);
            maxCenterNoise = radius * (3.5 * (1-pressure)*(1-pressure) + 0.25);
        }
    };
    updatePressureParams();

    auto drawCircle = [&](const Point2D& center) {
        if(_rgb) {
            drawSoftCircle(_noise2D.addNoise(center, maxCenterNoise), radius, greatestPixValChange, pressure, _pixTraitsRGB);
        }
        else {
            drawSoftCircle(_noise2D.addNoise(center, maxCenterNoise), radius, greatestPixValChange, pressure, PixTraitsAlpha());
        }
    };

    // Compute segment length and direction
    double segmentLength = distance(start, end);
    Point2D direction = diff(end, start);

    if (segmentLength == 0) {
        if (_isFirstSegment) {
            drawCircle(start);
            _lastCircleCenter = start;
            _remainingDistance = interval;
        }
        return;
    }

    // Compute unit direction
    Point2D unitDirection = mul(direction, 1.0 / segmentLength);

    // For the first segment, start at the start point
    if (_isFirstSegment) {
        drawCircle(start);
        _lastCircleCenter = start;
        _remainingDistance = interval;
        _isFirstSegment = false;
    }

    // Draw circles at intervals, starting from _remainingDistance
    double distanceToNext = _remainingDistance;
    while (distanceToNext <= segmentLength + 1e-10) { // Small epsilon for floating-point errors
        Point2D center = sum(start, mul(unitDirection, distanceToNext));
        pressure = std::max(0.0, std::min(1.0, ((distanceToNext * ends.pressure + (segmentLength - distanceToNext)*starts.pressure)/segmentLength)));
        updatePressureParams();
        drawCircle(center);
        _lastCircleCenter = center;
        distanceToNext += interval;
    }

    // Update remaining distance to the next circle
    _remainingDistance = distanceToNext - segmentLength;

    onLinePainted(start, end);
}

void ImagePainter::resetROI()
{
    _xmin = 1000000;
    _ymin = 1000000;
    _xmax = -1;
    _ymax = -1;
}

void ImagePainter::getROI(int* region)
{
    region[0] = _xmin;
    region[1] = _ymin;
    region[2] = _xmax;
    region[3] = _ymax;
}

void ImagePainter::paintWithin(const int* region)
{
    const int entireImage[] = {0, 0, _width, _height};
    if(!region) {
        region = entireImage;
    }
    _minPainted.x = region[0];
    _minPainted.y = region[1];
    _maxPainted.x = region[2];
    _maxPainted.y = region[3];
}

void ImagePainter::drawLine(const Point2D& start, const Point2D& end, double width, const unsigned char* rgb)
{
    double raw_u = 0;
    auto projOntoLine = projectPointOntoLineSegument(start, end);

    int startx = std::max(floor(std::min(start.x,end.x) - width), _minPainted.x);
    int starty = std::max(floor(std::min(start.y,end.y) - width), _minPainted.y);
    int endx = std::min(ceil(std::max(start.x,end.x) + width), _maxPainted.x);
    int endy = std::min(ceil(std::max(start.y,end.y) + width), _maxPainted.y);
    double halfWidth = width * 0.5;

    double w = 2; //empirically, things get grainy below w=2 (say at 1.5), and at w=2,
    //the narrowest line width where we're guaranteed to have 255s all along it
    //for a 4-connected flood fill to be stopped by the line is 2.5

    //we can't erase an already fully-erased pixel, or strengthen an already-strongest-possible one
    int immutableVal = _erase ? 0 : 255;

    //detect sharp turns to prevent artifacts (see also below)
    if(!_detectSharpTurns) {
        _cumDist += distance(start, end);
        if(_cumDist > 15) {
            _detectSharpTurns = true;
        }
    }

    bool sharpTurn = false;
    if(!_lastStartValid) {
        _lastStartValid = true;
    }
    else if(_detectSharpTurns && std::max(distance(_lastStart, start), distance(start, end)) > 0) {
        double cos = cosOfAngleBetween2Lines(_lastStart, start, end);
        sharpTurn = cos < 0.5;
    }
    _lastStart = start;

    _aroundCurrEndpoint.setRegion(end, halfWidth+w+2);

    for(int y = starty; y < endy; y++) {
        for(int x = startx; x < endx; x++) {
            int ind = y*_ystride + x*_xstride;
            int oldVal = _image[ind];
            if(oldVal == immutableVal) {
                continue;
            }
            Point2D p{double(x),double(y)};
            double dist = distance(p, projOntoLine(p, raw_u));

            if(dist > halfWidth+w+1) {
                continue;
            }

            int grey = 255;
            double c = std::max(-w, std::min(w, dist - halfWidth));
            grey = (1-sigmoid(c*6/w)) * 255;
            if(grey >= 255 - 30) {
                grey = 255;
            }
            int newVal;
            if(_erase) {
                newVal = std::min(255-grey, oldVal);
            }
            //here we use alpha blending ("over") rather than max blending.
            //there's no reason to not do this for erasers as well, except that currently
            //the actually used erasers don't use drawLine but rather are pencil/soft circle
            //based - if we ever want to use "sharp" erasers, there's value in avoiding aliasing
            //artifacts when 2 adjacent lines (whether created by pens or erasers) get close enough
            //("over" blending makes these artifacts much less pronounced)
            else {
                int valAfterLastDrawLine = 0;
                CoordSet::Value v = _aroundPrevEndpoint.findPixel(x,y);
                if(v.valid && !sharpTurn) {
                    valAfterLastDrawLine = oldVal;
                    oldVal = v.pixVal; //even older than the original oldVal... this is the value
                    //_before_ the last drawLine. basically when hitting a pixel near the last end
                    //point, we take the maximal value between what the 2 lines would have painted
                    //there [which is similar to "reverting to max blending" at this region but
                    //not exactly since the stuff painted there /before the previous drawLine call/
                    //is blended with "over" rather than max]
                    //
                    //if we don't handle this overlap between the 2 line segments specially,
                    //we will see artifacts (fat dots) where they connect. we avoid this special
                    //casing upon sharp turns since it comes with an artifact of its own ("thin"
                    //lines at the turn, compared to the fatter lines further away from the turning
                    //point that result from over blending 2 lines drawn one on top of another)
                }
                newVal = (grey*255 + (255-grey)*oldVal + (1<<7)) >> 8;
                newVal = std::max(newVal, valAfterLastDrawLine);

                //if we wanted to paint a pixel stopping flood fill, let's do it regardless
                //of what came before us
                if(grey >= 255-30) {
                    newVal = 255;
                }
                if(distance(p, end) <= halfWidth+w+1) {
                    _aroundCurrEndpoint.addPixel(x,y,oldVal);
                }
            }
            if(newVal != oldVal) {
                _image[ind] = newVal;
                _xmin = std::min(_xmin, x);
                _ymin = std::min(_ymin, y);
                _xmax = std::max(_xmax, x);
                _ymax = std::max(_ymax, y);
                onPixelPainted(x, y, _image[ind]);
            }
        }
    }
    //should std::move the vector of CoordSet::Value without reallocation
    std::swap(_aroundCurrEndpoint, _aroundPrevEndpoint);

    onLinePainted(start, end);
}

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
    bool _softLines = false;

    //output image
    ImagePainter* _painter = nullptr;

    void initPaint(const SamplePoint& p);
    //smoothing modifies p
    void paint(SamplePoint& p, double zoomCoeff);
    void endPaint();

    const std::vector<SamplePoint>& polyline() const { return _polyline; }
    const std::vector<int>& sample2polyline() const { return _sample2polyline; }
  private:
    void paintAt(const SamplePoint& p);
    void paintLine(const SamplePoint& p1, const SamplePoint& p2);
    void paintBezierSegment(const SamplePoint& p1, const SamplePoint& p2, const Point2D& tangent1, const Point2D& tangent2);
    void paintBezierCurve(const SamplePoint& p1, const SamplePoint& p2, const Point2D& control1, const Point2D& control2);

    //painting state
    SamplePoint _prevP; //previous value of p passed to paint (or initPaint the first time paint is called)
    SamplePoint _olderP; //previous value of _prevP
    std::vector<SamplePoint> _history;
    std::vector<double> _distHist;
    bool _paintedAtLeastOnce = false;
    bool _haveTangent = false;
    Point2D _prevTangent{0, 0};
    std::vector<SamplePoint> _polyline; //these are the coordinates of the polyline that was painted - after Bezier smoothing etc.
    std::vector<int> _sample2polyline; //maps input sample indexes to the polyline index corresponding to the start of painting at that point
    inline void addSample2Polyline() { _sample2polyline.push_back(_polyline.size()); }
    int _sampleIndex = 0;
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
    //this is inaccurate since painting "lags" the arrival of new samples. however this is very simple
    //and isn't prone to worse errors than this where we push too many or too few entries to _sample2polyline
    addSample2Polyline();
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
                //not sure why Krita doesn't have this test; it prevents us from painting a line early on when we don't have
                //enough history to paint a bezier segment (rather than not having enough _curvature_ for it to make sense).
                //without this test it seems that we paint a line and then a bezier segment starting at the same point
                if(_olderP.pos != _prevP.pos) {
                    //printf("paintLine prevP %f %f -> p %f %f (olderP %f %f)\n", _prevP.pos.x, _prevP.pos.y, p.pos.x, p.pos.y, _olderP.pos.x, _olderP.pos.y);
                    paintLine(_prevP, p);
                }
            }
            else {
                //printf("paintBezierSegment %f %f -> %f %f\n", _olderP.pos.x, _olderP.pos.y, _prevP.pos.x, _prevP.pos.y);
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
    _painter->detectSharpTurns(false); //not sure this helps (with a fat dot appearing at the end in drawLine) but seemingly can't hurt
    addSample2Polyline();
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

const double BEZIER_FLATNESS_THRESHOLD = 0.1; //Krita uses 0.5; smaller is smoother and potentially more expensive

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
    bool repeatedLine = _polyline.size() == 2 && _polyline[0] == p1 && _polyline[1] == p2;
    if(!repeatedLine) {
        if(_polyline.empty()) {
            _polyline.push_back(p1);
            _polyline.push_back(p2);
        }
        else {
            if(_polyline.back() != p1) {
                //hopefully this no longer happens, and the reason it used to was that paintLine() was called early on because
                //we didn't have enough history to call paintBezierSegment() and then we called paintBezierSegment, repainting
                //from the same starting point as in the first paintLine() call
                printf("Brush::paintLine - WARNING: paintLine called with p1 different from the previous paintLine's p2!\n");
            }
            _polyline.push_back(p2);
        }
    }
    else {
        printf("Brush::paintLine - WARNING: repeated line!\n");
    }

    _paintedAtLeastOnce = true;
    if(_painter) {
        if(_softLines) {
          _painter->drawLineUsingWideSoftCiclesWithNoisyCenters(p1, p2, _lineWidth);
        }
        else {
          _painter->drawLine(p1.pos, p2.pos, _lineWidth);
        }
    }
}

void Brush::paintAt(const SamplePoint& p)
{
    SamplePoint peps = p;
    peps.pos.x += 0.001;
    paintLine(p, peps);
}

//extern "C" API

extern "C" Brush* brush_init_paint(double x, double y, double time, double pressure, double lineWidth, double smoothDist, int dry, int erase, int softLines,
                                   unsigned char* image, int width, int height, int xstride, int ystride, const int* paintWithinRegion)
{
    Brush& brush = *new Brush;
    brush._smoothing = Smoothing::WEIGHTED;
    brush._lineWidth = lineWidth;
    brush._smoothDist = smoothDist;
    brush._tailAggressiveness = 0; //TODO: for soft lines/pencil, might want to experiment with this parameter
    brush._softLines = softLines;
    
    if(!dry) {
        ImagePainter& painter = *new ImagePainter;
        painter._image = image;
        painter._width = width;
        painter._height = height;
        painter._xstride = xstride;
        painter._ystride = ystride;
        painter._erase = erase;
        painter.paintWithin(paintWithinRegion);
        brush._painter = &painter;
    }
    else {
        brush._painter = nullptr;
    }

    brush.initPaint({{x,y},time,pressure});

    return &brush;
}

//note that you should have created the brush pointing to rgb pixels if you are to call this function
//(by default you point to the alpha channel which is imageBase+3 in an RGB image, here you want
//to point to imageBase)
extern "C" void brush_set_rgb(Brush* brush, const unsigned char* rgb)
{
    ImagePainter& painter = *brush->_painter;
    painter._rgb = true;
    painter._pixTraitsRGB.newColor.rgb[0] = rgb[0];
    painter._pixTraitsRGB.newColor.rgb[1] = rgb[1];
    painter._pixTraitsRGB.newColor.rgb[2] = rgb[2];
}

extern "C" void brush_paint(Brush* brush, int npoints, double* x, double* y, const double* time, const double* pressure, double zoom, int* region)
{
    if(brush->_painter) {
        brush->_painter->resetROI();
    }
    for(int i=0; i<npoints; ++i) {
        SamplePoint s{{x[i],y[i]},time ? time[i] : (i+1)*7, pressure ? pressure[i] : 1};
        brush->paint(s, zoom);
        x[i] = s.pos.x;
        y[i] = s.pos.y;
    }
    if(brush->_painter) {
        brush->_painter->getROI(region);
    }
}

extern "C" void brush_end_paint(Brush* brush, int* region)
{
    if(brush->_painter) {
        brush->_painter->resetROI();
    }
    brush->endPaint();
    if(brush->_painter) {
        brush->_painter->getROI(region);
    }
}

extern "C" int brush_get_polyline_length(Brush* brush)
{
    return brush->polyline().size();
}

extern "C" void brush_get_polyline(Brush* brush, int polyline_length, double* polyline_x, double* polyline_y, double* polyline_time, double* polyline_pressure)
{
    const auto& polyline = brush->polyline();
    if(polyline_length != (int)polyline.size()) {
        printf("brush_get_polyline_and_free() - WARNING: wrong polyline length\n");
        return;
    }
    for(int i=0; i<polyline_length; ++i) {
        if(polyline_x) {
            polyline_x[i] = polyline[i].pos.x; 
        }
        if(polyline_y) {
            polyline_y[i] = polyline[i].pos.y; 
        }
        if(polyline_time) {
            polyline_time[i] = polyline[i].time; 
        }
        if(polyline_pressure) {
            polyline_pressure[i] = polyline[i].pressure; 
        }
    }
}

extern "C" void brush_get_sample2polyline(Brush* brush, int sample2polyline_length, int* sample2polyline_data)
{
    const auto& sample2polyline = brush->sample2polyline();
    if(sample2polyline_length != (int)sample2polyline.size()) {
        printf("brush_get_sample2polyline() - WARNING: wrong sample2polyline length\n");
        return;
    }
    std::copy(sample2polyline.begin(), sample2polyline.end(), sample2polyline_data);
}

extern "C" void brush_free(Brush* brush)
{
    delete brush->_painter;
    delete brush;
}

//flood-filling color as the lines are erased

extern "C" void flood_fill_color_based_on_mask_many_seeds(int* color, unsigned char* mask,
                int color_stride, int mask_stride, int width, int height,
                int* region, int _8_connectivity,
                int mask_new_val, int new_color_value,
                const int* seed_x, const int* seed_y, int num_seeds);

class FloodFillingPainter : public ImagePainter
{
  public:
    int* _color;
    unsigned char* _mask;
    int _color_stride;
    int _mask_stride;
    int _8_connectivity;
    int _mask_new_val;
    int _new_color_value;
    
    std::vector<int> _seeds_x;
    std::vector<int> _seeds_y;

    void onPixelPainted(int x, int y, int value) override
    {
        if(value < 255) {
            _mask[_mask_stride*y + x] = 0;
            _seeds_x.push_back(x);
            _seeds_y.push_back(y);
        }
    }
    void onLinePainted(const Point2D& start, const Point2D& end) override
    {
        if(_seeds_x.empty()) {
            return;
        }
        //theoretically just one of the points should suffice...
        int region[4];
        flood_fill_color_based_on_mask_many_seeds(_color, _mask, _color_stride, _mask_stride, _width, _height, region, _8_connectivity, _mask_new_val, _new_color_value,
                &_seeds_x[0], &_seeds_y[0], _seeds_x.size());
        if(region[3] >= 0) {
            _xmin = std::min(region[0], _xmin);
            _ymin = std::min(region[1], _ymin);
            //here, max values are inclusive
            _xmax = std::max(region[2]-1, _xmax);
            _ymax = std::max(region[3]-1, _ymax);
        }
        _seeds_x.clear();
        _seeds_y.clear();
    }
};

extern "C" void brush_flood_fill_color_based_on_mask(Brush* brush, int* color, unsigned char* mask,
        int color_stride, int mask_stride, int _8_connectivity, int mask_new_val, int new_color_value)
{
    FloodFillingPainter* painter = new FloodFillingPainter;
    painter->_image = brush->_painter->_image;
    painter->_width = brush->_painter->_width;
    painter->_height = brush->_painter->_height;
    painter->_xstride = brush->_painter->_xstride;
    painter->_ystride = brush->_painter->_ystride;
    painter->_erase = brush->_painter->_erase;

    painter->_color = color;
    painter->_mask = mask;
    painter->_color_stride = color_stride;
    painter->_mask_stride = mask_stride;
    painter->_8_connectivity = _8_connectivity;
    painter->_mask_new_val = mask_new_val;
    painter->_new_color_value = new_color_value;

    //we could have copied the region from the original painter but paintWithin(nonNullRegion)
    //doesn't make a lot of sense with flood filling
    painter->paintWithin(nullptr);

    delete brush->_painter;
    brush->_painter = painter;
}
