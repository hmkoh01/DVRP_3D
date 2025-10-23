import math
from ..models.entities import Position

def vector(a, b):
    return Position(b[0]-a[0], b[1]-a[1])

def ccw(A, B, C):
    a, b = vector(A, B), vector(A, C)
    return a[0]*b[1] - a[1]*b[0]

def segments_intersect(l1, l2):
    '''
    not cross : 0
    end point cross : 1
    overlap : 2
    cross : 3
    '''
    A, B = l1
    C, D = l2
    if A > B: A, B = B, A
    if C > D: C, D = D, C
    abc = ccw(A, B, C)
    abd = ccw(A, B, D)
    cda = ccw(C, D, A)
    cdb = ccw(C, D, B)
    x = abc*abd
    y = cda*cdb
    if abc == abd == 0:
        if B < C or D < A: return 0
        elif B == C or D == A: return 1
        else: return 2
    elif x <= 0 and y <= 0:
        if x == 0 or y == 0:
            return 1
        else:
            return 3
    else: return 0

def segment_intersects_polygon(line, polygon):
    '''
    not cross : 0
    touch : 1
    overlap : 2
    intersect : 3
    '''
    result = 0
    for i in range(len(polygon)):
        result = max(segments_intersect(line, (polygon[i-1], polygon[i])), result)
    return result

def is_in_polygon(polygon, p):
    count = 0
    v = (p, (int(1e9), p[1]+1))
    for index in range(len(polygon)):
        line = (polygon[index-1], polygon[index])
        if segments_intersect(line, v): count += 1
    return count % 2 == 1

def dist(a, b):
    v = vector(a, b)
    return math.hypot(v[0], v[1])