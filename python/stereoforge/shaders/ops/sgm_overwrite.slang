// Experimental first path; kept in sync with frozen specialized recurrence.
#include "sgm_specialized.slang"
void aggregate_overwrite_body<let Components:uint,let Bits:uint,let RightReference:bool>(Parameters parameters,uint path,uint groups_x,uint3 gid){
    uint width=parameters.width,height=parameters.height,disparities=parameters.disparities,p1=parameters.p1,p2=parameters.p2;
    uint* left_desc=(uint*)(RightReference?parameters.right_desc_address:parameters.left_desc_address);uint* right_desc=(uint*)(RightReference?parameters.left_desc_address:parameters.right_desc_address);
    uint* left_valid=(uint*)(RightReference?parameters.right_valid_address:parameters.left_valid_address);uint* right_valid=(uint*)(RightReference?parameters.left_valid_address:parameters.right_valid_address);
    uint* words=(uint*)parameters.penalties_address;
    uint lanes=WaveGetLaneCount(),lane=WaveGetLaneIndex();int2 dir=sgm_direction(path);
    uint line=(gid.y*groups_x+gid.x)*(64/lanes)+WaveGetWaveIndex();if(line>=sgm_line_count(width,height,dir))return;
    uint row_words=(disparities+Components-1)/Components,chunks=(row_words+lanes-1)/lanes;
    uint prev[8],next[8];[unroll]for(uint r=0;r<8;++r){prev[r]=0;next[r]=0;}
    int2 xy=sgm_line_start(width,height,dir,line);bool first=true;
    while(sgm_inside(xy,width,height)){
        uint local_min=SGM_REDUCTION_INF;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){[unroll]for(uint component=0;component<Components;++component)if((k*lanes+lane)*Components+component<disparities)local_min=min(local_min,prev[k*Components+component]);}
        uint minimum=WaveActiveMin(local_min),x=uint(xy.x),i=uint(xy.y)*width+x;
        uint reference=left_desc[i];bool reference_valid=left_valid[i]!=0;
        [unroll]for(uint k=0;k<8/Components;++k)if(k<chunks){
            // Full-wave uniform shuffles occur even for lanes whose word/tail is masked.
            uint lo_same=WaveReadLaneAt(prev[k*Components+Components-1],(lane+lanes-1)%lanes);
            uint hi_same=WaveReadLaneAt(prev[k*Components],(lane+1)%lanes);
            uint lo_cross=WaveReadLaneAt(k>0?prev[(k>0?k-1:0)*Components+Components-1]:SGM_REDUCTION_INF,lanes-1);
            uint hi_cross=WaveReadLaneAt(k+1<chunks?prev[(k+1<8/Components?k+1:k)*Components]:SGM_REDUCTION_INF,0);
            uint word=k*lanes+lane,addition=0;
            [unroll]for(uint component=0;component<Components;++component){uint d=word*Components+component,r=k*Components+component;
                uint lo=component>0?prev[r>0?r-1:0]:(lane>0?lo_same:lo_cross);
                uint hi=component+1<Components?prev[min(r+1,7u)]:(lane+1<lanes?hi_same:hi_cross);
                if(d==0)lo=SGM_REDUCTION_INF;if(d+1>=disparities)hi=SGM_REDUCTION_INF;
                if(d<disparities){uint c=reference_cost<RightReference>(width,x,i,d,reference,reference_valid,right_valid,right_desc);next[r]=sgm_step(c,prev[r],lo,hi,minimum,p1,p2,first);addition|=(next[r]-c)<<(component*Bits);}else next[r]=0;
            }
            if(word<row_words)words[i*row_words+word]=addition;
        }
        [unroll]for(uint r=0;r<8;++r)prev[r]=next[r];first=false;xy+=dir;
    }
}
