inline float fast_sqrt(float x) {
    return x * rsqrt_fast(x);
}

/* a curve segment is the shape rendered by Brush::drawLine - a trapese with 2 half-circles
   defined by a line and 2 widths which define the radii of the 2 half-circles

   projectPointOntoLineSegment and its use in drawLine compute the distance field in brush.cpp
*/
export void point_to_curve_segments_distances(uniform int nsegments, uniform float distances[],
    uniform const float xstarts[], uniform const float ystarts[], uniform const float wstarts[],
    uniform const float xends[], uniform const float yends[], uniform const float wends[], 
    uniform float x, uniform float y)
{
    foreach(i=0 ... nsegments) {
        float sx = xstarts[i];
        float sy = ystarts[i];
        float sw = wstarts[i];
        float ex = xends[i];
        float ey = yends[i];
        float ew = wends[i];
        
        float dx = ex-sx;
        float dy = ey-sy;
        float sqLineMag = dx*dx + dy*dy;
        float invSqLineMag = rcp_fast(sqLineMag);

        float pdx = x-sx;
        float pdy = y-sy;

        float raw_u = (pdx*dx + pdy*dy) * invSqLineMag;
        float u = min(1.0f, max(0.0f, raw_u));

        float projx = sx + dx*u;
        float projy = sy + dy*u;

        float width = select(raw_u <= 0.0f, sw, select(raw_u >= 1.0f, ew, raw_u*ew + (1.0f-raw_u)*sw)); 
        float dist = fast_sqrt((projx-x)*(projx-x) + (projy-y)*(projy-y));
        float distToLine = max(0.0f, dist - width*0.5f);

        //for the redundant case where start==end:
        float averageWidth = (sw+ew)*0.5f;
        float distToStart = max(0.0f, fast_sqrt(pdx*pdx + pdy*pdy) - averageWidth*0.5f);

        distances[i] = select(sqLineMag < 1e-8, distToStart, distToLine); 
    }
}
