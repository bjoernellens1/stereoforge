// Positive d=x_left-x_right; symmetric LR, no confidence threshold.
bool observation_target(uint width,uint x,float d,bool right,out uint target){
    target=0;if(!isfinite(d)||abs(d)>float(width+1))return false;
    // Exact integer translation of floor(x +/- binary32(d) + 0.5).
    // Compare against the exact half integer; d-floor(d) can lose negative ties.
    int base=int(floor(d));float threshold=float(base)+0.5f;
    int delta=base+int(right?d>=threshold:d>threshold);
    int coordinate=int(x)+(right?delta:-delta);
    if(coordinate<0||coordinate>=int(width))return false;
    target=uint(coordinate);return true;
}
bool observation_lr(float own,float other,uint own_valid,uint other_valid,float tolerance){return own_valid!=0&&other_valid!=0&&isfinite(own)&&isfinite(other)&&abs(own-other)<=tolerance;}
float observation_z(float disparity,float fx,float baseline,float principal_offset){return (fx*baseline)/(disparity-principal_offset);}
