#![allow(non_snake_case)]

fn apply_xy2(N:usize,grid:&mut [f64],oi:usize,oj:usize){
    let r0=rnd::rangef(0.001,0.1);
    let r1=rnd::rangef(0.001,0.1);
    let amp=rnd::rangef(10.,100.);
    let mut k=0;
    for i in 0..N{
        for j in 0..N{
            let di=(i-oi)*(i-oi);
            let dj=(j-oj)*(j-oj);
            grid[k]+=amp*(-(di as f64*r0+dj as f64*r1)).exp();
            k+=1;
        }
    }
}


fn normalize(N:usize,grid:&mut [f64]){
    let vmin=*grid.iter().min_by(|a,b|a.partial_cmp(b).unwrap()).unwrap();
    let vmax=*grid.iter().max_by(|a,b|a.partial_cmp(b).unwrap()).unwrap();
    let inv=1./(vmax-vmin);
    for i in 0..N*N{
        grid[i]=(grid[i]-vmin)*inv;
    }
}


fn generate(N:usize,grid:&mut [f64]){
    for _ in 0..rnd::range(MINE,MAXE+1){
        let oi=rnd::next()%(N+1);
        let oj=rnd::next()%(N+1);
        apply_xy2(N,grid,oi,oj);
    }

    normalize(N,grid);
}


const DIJ:[(usize,usize);9]=[(!0,!0),(!0,0),(!0,1),(0,!0),(0,0),(0,1),(1,!0),(1,0),(1,1)];


fn hide(N:usize,ans:&[f64],sampled:&mut [f64],flag:&mut [usize],bf:&mut Vec<(usize,usize,f64)>){
    bf.clear();
    let mut it=0;

    // todo
    let MINH=(N as f64*N as f64/150.).round() as usize;
    let MAXH=(N as f64*N as f64/50.).round() as usize;
    
    for _ in 0..rnd::range(MINH,MAXH+1){
        let (mut i,mut j);
        loop{
            i=rnd::range(1,N-1);
            j=rnd::range(1,N-1);
            if !DIJ.into_iter().any(|(di,dj)|flag[(i+di)*N+j+dj]!=0){
                break;
            }
        }
        it+=1;

        for (di,dj) in DIJ{
            let ni=i+di;
            let nj=j+dj;
            flag[ni*N+nj]=it;

            bf.push((ni,nj,ans[ni*N+nj]));
        }
    }

    for i in 0..N{
        for j in 0..N{
            let &(_,_,v0)=bf.iter().min_by_key(|(ai,aj,_)|(ai-i)*(ai-i)+(aj-j)*(aj-j)).unwrap();
            sampled[i*N+j]=v0;
        }
    }
}



const MINE:usize=2;
const MAXE:usize=5;
const COUNT:usize=10000;


fn main(){
    for N in [24,32,40,48,56,64]{
        let mut ans=vec![0.;N*N];
        let mut flag=vec![0;N*N];
        let mut bf=vec![];
        let mut sampled=vec![0.;N*N];

        let data_dir=format!("data_only/data_{N}");

        for i in 0..COUNT{
            ans.fill(0.);
            flag.fill(0);
            sampled.fill(0.);

            generate(N,&mut ans);
            hide(N,&ans,&mut sampled,&mut flag,&mut bf);

            let mut file=File::create(format!("{data_dir}/ans/{i:0>4}.txt")).unwrap();
            for i in 0..N*N{
                assert!(0.<=ans[i] && ans[i]<=1.);
                write!(file,"{:.4} ",ans[i]).unwrap();
            }
            file.flush().unwrap();

            let mut file=File::create(format!("{data_dir}/sampled/{i:0>4}.txt")).unwrap();
            for i in 0..N*N{
                assert!(0.<=sampled[i] && sampled[i]<=1.);
                write!(file,"{:.4} ",sampled[i]).unwrap();
            }
            file.flush().unwrap();

            let mut file=File::create(format!("{data_dir}/flag/{i:0>4}.txt")).unwrap();
            for i in 0..N*N{
                write!(file,"{} ",(flag[i]>0) as usize).unwrap();
            }
            file.flush().unwrap();
        }

        eprintln!("finished: {}",N);
    }
}


use std::fs::File;
use std::io::Write;


mod rnd {
    static mut S:usize=88172645463325252;

    pub fn next()->usize{
        unsafe{
            S^=S<<7;
            S^=S>>9;
            S
        }
    }

    pub fn nextf()->f64{
        unsafe{
            std::mem::transmute::<u64,f64>((0x3ff0000000000000|next()&0xfffffffffffff) as u64)-1.
        }
    }

    pub fn range(a:usize,b:usize)->usize{
        next()%(b-a)+a
    }

    pub fn rangef(a:f64,b:f64)->f64{
        nextf()*(b-a)+a
    }
}
