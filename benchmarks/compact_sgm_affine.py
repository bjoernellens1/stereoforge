"""CPU-only affine photometric ablation using frozen Census LR anchors."""
import numpy as np

PARAMETERS={'tile':64,'per_tile':64,'train_tile_residues':[0,1,2],'validation_tile_residues':[3], 'min_train':1024,'min_validation':256,'min_train_tiles':16,'min_validation_tiles':4,'min_span':32,'refit_count':3,'residual_threshold':12,'min_inliers':512,'gain_min':.75,'gain_max':1.33,'offset_limit':40,'validation_median_limit':8,'validation_p90_limit':20}

def sample_anchors(left,right,left_support,right_support,disparity,lr_valid):
    left,right=np.asarray(left),np.asarray(right);left_support=np.asarray(left_support,dtype=bool);right_support=np.asarray(right_support,dtype=bool);valid=np.asarray(lr_valid,dtype=bool)
    if left.dtype!=np.uint8 or right.dtype!=np.uint8 or any(x.shape!=left.shape for x in (right,left_support,right_support,disparity,valid)) or left.ndim!=2:raise ValueError('invalid anchor inputs')
    h,w=left.shape;y,x=np.indices((h,w));d=np.asarray(disparity);finite=np.isfinite(d);di=np.where(finite,d,0).astype(np.int32);ox=x-di
    if np.any(valid & (~finite | (d!=di) | (di<0) | (ox<0) | (ox>=w) | ~left_support)):raise ValueError('invalid anchor reprojection')
    target=np.clip(ox,0,w-1);rv=right[y,target]
    if np.any(valid & ~right_support[y,target]):raise ValueError('unsupported target anchor')
    eligible=valid&(left>0)&(left<255)&(rv>0)&(rv<255)
    out={}
    for label,residues in [('train',(0,1,2)),('validation',(3,))]:
        ys=[];xs=[];tiles=[]
        for ty in range((h+63)//64):
            for tx in range((w+63)//64):
                if (tx+3*ty)%4 not in residues:continue
                sub=eligible[ty*64:min((ty+1)*64,h),tx*64:min((tx+1)*64,w)]
                a,b=np.nonzero(sub)
                if len(a):
                    ys.extend((a[:64]+ty*64).tolist());xs.extend((b[:64]+tx*64).tolist());tiles.append((ty,tx))
        yy=np.asarray(ys,dtype=np.int32);xx=np.asarray(xs,dtype=np.int32)
        out[label+'_left']=left[yy,xx].astype(np.float64);out[label+'_right']=right[yy,xx-di[yy,xx]].astype(np.float64)
        out[label+'_x']=xx;out[label+'_y']=yy;out[label+'_tiles']=tiles
    return out

def nearest(v,p):
    if len(v)==0:raise ValueError('empty percentile')
    return float(np.partition(v,max(0,int(np.ceil(p*len(v)))-1))[max(0,int(np.ceil(p*len(v)))-1)])

def estimate_affine(anchors):
    x,y=anchors['train_right'],anchors['train_left'];vx,vy=anchors['validation_right'],anchors['validation_left']
    result={'accepted':False,'reason':'','train_count':len(x),'validation_count':len(vx),'train_tiles':len(anchors['train_tiles']),'validation_tiles':len(anchors['validation_tiles']),'inliers':0}
    def reject(reason):result['reason']=reason;return result
    if len(x)<1024 or len(vx)<256 or result['train_tiles']<16 or result['validation_tiles']<4:return reject('insufficient coverage')
    if nearest(x,.9)-nearest(x,.1)<32 or nearest(vx,.9)-nearest(vx,.1)<32:return reject('insufficient intensity spread')
    current=np.ones(len(x),bool)
    for step in range(4):
        xx=x[current];yy=y[current]
        if len(xx)<512:return reject('insufficient inliers')
        center=xx.mean();denom=np.sum((xx-center)**2)
        if denom<=0:return reject('degenerate fit')
        gain=float(np.sum((xx-center)*(yy-yy.mean()))/denom);offset=float(yy.mean()-gain*center)
        if step<3:current=np.abs(y-(gain*x+offset))<=12
    result.update({'gain':gain,'offset':offset,'inliers':int(current.sum())})
    if not .75<=gain<=1.33 or abs(offset)>40:return reject('gain or offset out of bounds')
    residual=np.abs(vy-(gain*vx+offset));result.update({'validation_median':nearest(residual,.5),'validation_p90':nearest(residual,.9)})
    if result['validation_median']>8 or result['validation_p90']>20:return reject('validation residual')
    result['accepted']=True;result['reason']='accepted';return result

def apply_affine(right,gain,offset):
    if right.dtype!=np.uint8 or not np.isfinite([gain,offset]).all():raise ValueError('invalid correction')
    mapped=np.rint(right.astype(np.float64)*gain+offset);safe=(right>0)&(right<255)&(mapped>0)&(mapped<255)
    return np.clip(mapped,0,255).astype(np.uint8),safe

def run_masked(runner,directory,own,other,image,other_image,safe,other_safe,lo,hi,eye,disparities):
    from pathlib import Path
    import subprocess
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);h,w=image.shape
    if eye not in (0,1) or image.dtype!=np.uint8 or other_image.dtype!=np.uint8 or safe.dtype!=bool or other_safe.dtype!=bool or any(a.shape!=(h,w) for a in (other_image,safe,other_safe,own.descriptor,other.descriptor,own.support,other.support,lo,hi)):raise ValueError('invalid input')
    for name,a in [('own',own.descriptor),('other',other.descriptor),('own_support',own.support),('other_support',other.support),('luma',image),('other_luma',other_image),('safe',safe),('other_safe',other_safe),('lo',lo),('hi',hi)]:np.asarray(a,dtype='<u4').tofile(root/name)
    subprocess.run([str(runner),str(root),str(h),str(w),str(disparities),'16','64',str(eye),'masked'],check=True)
    result={name:np.fromfile(root/name,dtype=dtype).reshape(h,w) for name,dtype in [('winners','<i4'),('best','<u4'),('second','<u4'),('count','<u4')]}
    result['candidate_updates'],result['interval_slots']=map(int,(root/'stats').read_text().split());return result


def fallback_selection(fit, baseline):
    """Declined correction must copy every frozen Census output plane exactly."""
    if fit.get("accepted") is not False:raise ValueError("fallback requires rejected fit")
    return {name:np.array(value,copy=True) for name,value in baseline.items()}
