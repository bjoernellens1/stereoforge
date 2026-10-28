"""Fail-closed structural SPIR-V check for the scoped subgroup shader."""
import struct
from pathlib import Path


def validate(data):
    if len(data)<20 or len(data)%4:raise ValueError('invalid SPIR-V length')
    words=struct.unpack('<%dI'%(len(data)//4),data)
    if words[0]!=0x07230203:raise ValueError('invalid SPIR-V magic')
    ops=[];index=5
    while index<len(words):
        size=words[index]>>16;op=words[index]&0xffff
        if not size or index+size>len(words):raise ValueError('malformed instruction')
        ops.append((op,words[index+1:index+size]));index+=size
    caps={x[0] for op,x in ops if op==17 and len(x)==1}
    constants={x[1]:x[2] for op,x in ops if op==43 and len(x)==3}
    workgroup_vars={x[1] for op,x in ops if op==59 and len(x)>=3 and x[2]==4}
    if not {61,63}<=caps:raise ValueError('missing subgroup capabilities')
    elect=[x for op,x in ops if op==333 and len(x)==3 and constants.get(x[2])==3]
    minimum=[x for op,x in ops if op==354 and len(x)==5 and constants.get(x[2])==3 and x[3]==0]
    atomic=[x for op,x in ops if op==237 and len(x)==6 and x[2] in workgroup_vars]
    barriers=[x for op,x in ops if op==224 and len(x)==3 and constants.get(x[0])==2 and constants.get(x[1])==2 and constants.get(x[2],0)&0x100]
    if len(elect)!=1 or len(minimum)!=1 or len(atomic)!=1 or len(barriers)<5:raise ValueError('missing subgroup reduction, shared atomic or workgroup barriers')
    return {'subgroup_elect':len(elect),'subgroup_umin':len(minimum),'workgroup_atomic_umin':len(atomic),'workgroup_barriers':len(barriers)}


if __name__=='__main__':
    import sys
    print(validate(Path(sys.argv[1]).read_bytes()))
