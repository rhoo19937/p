#![allow(non_snake_case)]


// #[cfg(local)]
// use io_local::*;
// #[cfg(not(local))]
use io_topcoder::*;


fn predict_grid(IO:&mut IO,unet:&UNet)->Vec<Vec<i64>>{
    let N=(IO.N+7)/8*8;
    
    let mut pos=vec![];
    let mut x=vec![vec![vec![0.;N];N];2];

    for p in iterp(IO.N){
        if let Some(v)=IO.get(p){
            pos.push((p,v));
            x[1][p]=1.;
        }
    }

    for p in iterp(N){
        let (_,v)=*pos.iter().min_by_key(|t|(t.0-p).abs2()).unwrap();
        x[0][p]=v as f32/255.;
    }

    let mut xt=vec![];
    for i in 0..2{
        for p in iterp(N){
            xt.push(x[i][p]);
        }
    }

    let y=unet.apply(&Array3::from_shape_vec([2,N,N],xt).unwrap());
    assert_eq!(y.shape()[0],1);

    let mut ret=vec![vec![0;IO.N];IO.N];
    for i in 0..IO.N{
        for j in 0..IO.N{
            ret[i][j]=(y[[0,i,j]]*255.).round().min(255.).max(0.) as i64;
        }
    }

    for i in 0..pos.len(){
        let d=IO.get(pos[i].0).unwrap()-ret[pos[i].0];
        pos[i].1=d;
    }

    for p in iterp(IO.N){
        if let Some(v)=IO.get(p){
            ret[p]=v;
        }
        else{
            let (np,d)=*pos.iter().min_by_key(|t|(t.0-p).abs2()).unwrap();
            let dist=((p-np).abs2() as f64).sqrt();
            assert!(dist!=0.);
            const R:f64=10.;
            if dist<=R{ // todo
                let t=(dist-1.)/(R-1.);
                let r=gerp(0.8,0.5,t.powf(2.));
                let new=ret[p]+(d as f64*r).round() as i64;
                ret[p]=new.min(255).max(0);
            }
        }
    }

    ret
}


fn score(IO:&IO,p:P,pos:&[P],idx:usize)->f64{
    let mut ret=(p.0.min(p.1).min(IO.N as i64-p.0).min(IO.N as i64-p.1)) as f64*2.9; // todo
    for i in 0..pos.len(){
        if i!=idx{
            ret=ret.min(((pos[i]-p).abs2() as f64).sqrt());
        }
    }
    ret
}


fn gen(n:i64)->P{
    loop{
        let p=P(rnd::rangei(-n,n+1),rnd::rangei(-n,n+1));
        if p!=P(0,0){
            return p;
        }
    }
}


fn hc(IO:&IO,pos:&mut Vec<P>,idx:usize){
    for i in 0..(pos.len()-idx)*1000{ // todo
        let i=i%(pos.len()-idx)+idx;
        let p=pos[i];
        let np=p+gen(3);
        if score(IO,np,pos,i)>=score(IO,p,pos,i){
            pos[i]=np;
        }
    }
}


fn get_points(IO:&IO)->Vec<P>{
    // todo
    let Q={
        let a=(IO.A-0.2)/0.6;
        let n=(IO.N as f64-20.)/40.;
        
        let T0=lerp(0.006,0.004,n.powf(0.7));
        let T1=0.012;
        
        (lerp(T0,T1,a.powf(1.5))*(IO.N as f64*IO.N as f64)).round().max(5.).min(100.) as usize
    };
    let csq=include!("data/csq.txt"); // http://www.packomania.com packings of equal circles in a square

    let pos=&csq[Q];
    let miny=pos.iter().map(|t|O(t.0)).min().unwrap().0;
    let maxy=pos.iter().map(|t|O(t.0)).max().unwrap().0;
    let minx=pos.iter().map(|t|O(t.1)).min().unwrap().0;
    let maxx=pos.iter().map(|t|O(t.1)).max().unwrap().0;
    let free=0.5/[miny,maxy,minx,maxx].into_iter().map(|v|O((v-0.5f64).abs())).max().unwrap().0;

    let t=(Q as f64-5.)/40.;
    let ratio=1.+(free-1.)*lerp(0.18,0.22,t); // todo
    
    pos.iter().map(|&(y,x)|{
        let ny=(((y-0.5)*ratio+0.5)*IO.N as f64).round() as i64;
        let nx=(((x-0.5)*ratio+0.5)*IO.N as f64).round() as i64;
        let p=P(ny,nx);
        assert!(p.in_range(IO.N));
        p
    }).collect()
}



fn get_std(IO:&IO)->f64{
    let mut data=vec![];
    for p in iterp(IO.N){
        let Some(v0)=IO.get(p) else{continue};
        for dd in [P(0,1),P(1,0)]{
            let np=p+dd;
            if np.in_range(IO.N){
                let Some(v1)=IO.get(np) else{continue};
                data.push(v1-v0);
            }
        }
    }

    let mean=data.iter().sum::<i64>() as f64/data.len() as f64;
    let var=data.iter().map(|&v|(v as f64-mean).powi(2)).sum::<f64>()/data.len() as f64;
    
    var.sqrt()
}


fn solve(IO:&mut IO){
    let mut pos=get_points(IO);
    pos.sort_unstable_by_key(|p|p.0+p.1);
    let mut f=false;
    let mut i=0;
    while i<pos.len(){
        IO.next(pos[i]);
        i+=1;
        if !f && pos.len() as f64*0.5<=i as f64{
            f=true;
            let std=get_std(IO);
            if std<=4.{
                rnd::shuffle(&mut pos[i..]);
                pos.truncate((pos.len() as f64*0.85).round() as usize);
                hc(IO,&mut pos,i);
            }
            else if std<=7.{
                rnd::shuffle(&mut pos[i..]);
                pos.truncate((pos.len() as f64*0.9).round() as usize);
                hc(IO,&mut pos,i);
            }
        }
    }

    let f=IO.history.iter().any(|&p|{
        let mut cnt=0;
        for i in -1..=1{
            for j in -1..=1{
                let np=p+P(i,j);
                cnt+=(IO.get(np).unwrap()<=1) as usize; // todo
            }
        }
        cnt==9
    });

    let unet=UNet::new(IO.N,f);
    let ans=predict_grid(IO,&unet);

    IO.finish(&ans);
}


fn main(){
    let mut IO=IO::new();
    solve(&mut IO);
}


use itertools::*;


fn iterp(N:usize)->impl Iterator<Item=P>{
    iproduct!(0..N,0..N).map(|(i,j)|P::new(i,j))
}

fn lerp(a:f64,b:f64,t:f64)->f64{
    a+(b-a)*t
}

fn gerp(a:f64,b:f64,t:f64)->f64{
    a*(b/a).powf(t)
}


#[allow(unused)]
mod io_local{
    use super::*;
    pub struct IO{
        pub N:usize,
        pub A:f64,
        E:usize,
        ans:Vec<Vec<i64>>,
        pub grid:Vec<Vec<i64>>,
        cnt:usize,
        pub history:Vec<P>,
        pub time:f64,
    }
    impl IO{
        pub fn new()->IO{
            let mut scan=Scanner::new();
            let N:usize=scan.read();
            let A:f64=scan.read();
            let E:usize=scan.read();
            std::eprintln!("N = {}",N);
            std::eprintln!("A = {}",A);
            std::eprintln!("E = {}",E);
            let mut ans=vec![vec![0;N];N];
            for p in iterp(N){
                ans[p]=scan.read();
            }
            
            IO{
                N,A,E,
                ans,
                grid:vec![vec![-1;N];N],
                cnt:0,
                history:vec![],
                time:0.,
            }
        }

        pub fn isok(&self,pos:P)->bool{
            (pos-P(1,1)).in_range(self.N-2)
            && iproduct!(-1..=1,-1..=1).all(|(i,j)|self.get(pos+P(i,j)).is_none())
        }

        pub fn get(&self,pos:P)->Option<i64>{
            let v=self.grid[pos];
            if v==-1{
                None
            } else {
                Some(v)
            }
        }
    
        pub fn next(&mut self,pos:P){
            self.history.push(pos);
            self.cnt+=1;
            for i in -1..=1{
                for j in -1..=1{
                    let np=pos+P(i,j);
                    assert!(self.get(np).is_none());
                    self.grid[np]=self.ans[np];
                }
            }
            
            self.time=get_time();
        }
    
        pub fn finish(&mut self,ans:&Vec<Vec<i64>>){
            let max=self.N*self.N/9;
            assert!(0<self.cnt && self.cnt<max);
            const MAX_MSE:f64=4096.;
            let mut mse=0.;
            for p in iterp(self.N){
                if let Some(v)=self.get(p){
                    assert_eq!(ans[p],v);
                }
                assert!(0<=ans[p] && ans[p]<256);
                mse+=(self.ans[p] as f64-ans[p] as f64).powi(2);
            }
            mse/=self.N as f64*self.N as f64;
    
            let score=self.A*mse/MAX_MSE+(1.-self.A)*self.cnt as f64/max as f64;
            std::eprintln!("mse = {:.2}",mse);
            std::eprintln!("sample = {}",self.cnt);
            std::eprintln!("score = {:.5}",score*1e5);
    
            std::process::exit(0);
        }
    }
}


#[allow(unused)]
mod io_topcoder{
    use super::*;
    pub struct IO{
        pub scan:Scanner,
        pub N:usize,
        pub A:f64,
        pub grid:Vec<Vec<i64>>,
        pub history:Vec<P>,
        pub time:f64,
    }
    impl IO{
        pub fn new()->IO{
            let mut scan=Scanner::new();
            let A:f64=scan.read();
            let N:usize=scan.read();
            IO{
                scan,N,A,
                grid:vec![vec![-1;N];N],
                history:vec![],
                time:0.,
            }
        }

        pub fn isok(&self,pos:P)->bool{
            (pos-P(1,1)).in_range(self.N-2)
            && iproduct!(-1..=1,-1..=1).all(|(i,j)|self.get(pos+P(i,j)).is_none())
        }

        pub fn get(&self,pos:P)->Option<i64>{
            let v=self.grid[pos];
            if v==-1{
                None
            } else {
                Some(v)
            }
        }
    
        pub fn next(&mut self,pos:P){
            self.history.push(pos);
            println!("{} {}",pos.0,pos.1);
            for i in -1..=1{
                for j in -1..=1{
                    let np=pos+P(i,j);
                    let v:i64=self.scan.read();
                    assert!(self.get(np).is_none());
                    self.grid[np]=v;
                }
            }
    
            self.time=self.scan.read();
        }
        
        pub fn finish(&mut self,ans:&Vec<Vec<i64>>){
            println!("done");
            for p in iterp(self.N){
                if let Some(v)=self.get(p){
                    assert_eq!(ans[p],v);
                }
                println!("{}",ans[p]);
            }
            std::process::exit(0);
        }
    }
}


#[macro_export]#[cfg(not(local))]macro_rules! eprint{($($_:tt)*)=>{}}
#[macro_export]#[cfg(not(local))]macro_rules! eprintln{($($_:tt)*)=>{}}


pub struct Scanner{
    stack:std::str::SplitAsciiWhitespace<'static>
}
impl Scanner{
    pub fn new()->Self{
        Self{stack:"".split_ascii_whitespace()}
    }

    pub fn read<T:std::str::FromStr>(&mut self)->T{
        loop{
            if let Some(v)=self.stack.next(){
                return v.parse::<T>().unwrap_or_else(|_|panic!("{}: parse error",std::any::type_name::<T>()));
            }

            let mut tmp=String::new();
            std::io::stdin().read_line(&mut tmp).unwrap();
            assert!(!tmp.is_empty());
            self.stack=Box::leak(tmp.into_boxed_str()).split_ascii_whitespace();
        }
    }
}


pub fn get_time()->f64{
    static mut START:f64=-1.;
    let time=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    unsafe{
        if START<0.{
            START=time;
        }

        #[cfg(local)]{
            (time-START)*1.6
        }
        #[cfg(not(local))]{
            time-START
        }
    }
}


#[macro_export]
macro_rules! timer{
    ()=>{
        let _timer=Timer(get_time());
    }
}

static mut TIME:f64=0.;
struct Timer(f64);
impl Drop for Timer{
    fn drop(&mut self){
        unsafe{
            TIME+=get_time()-self.0
        }
    }
}


#[derive(Clone,Copy,PartialEq,Eq,Debug,Default,Hash)]
pub struct P(i64,i64);
impl P{
    fn new(a:usize,b:usize)->P{
        P(a as i64,b as i64)
    }
    
    fn in_range(self,n:usize)->bool{
        n>self.0 as usize && n>self.1 as usize
    }

    fn abs2(self)->i64{
        self.0*self.0+self.1*self.1
    }
}
use std::ops::*;
impl Add for P{
    type Output=P;
    fn add(self,a:P)->P{
        P(self.0+a.0,self.1+a.1)
    }
}
impl Sub for P{
    type Output=P;
    fn sub(self,a:P)->P{
        P(self.0-a.0,self.1-a.1)
    }
}
impl Mul<i64> for P{
    type Output=P;
    fn mul(self,a:i64)->P{
        P(self.0*a,self.1*a)
    }
}
impl Div<i64> for P{
    type Output=P;
    fn div(self,a:i64)->P{
        P(self.0/a,self.1/a)
    }
}
impl Neg for P{
    type Output=P;
    fn neg(self)->P{
        P(-self.0,-self.1)
    }
}
impl AddAssign for P{
    fn add_assign(&mut self,a:P){
        *self=*self+a;
    }
}
impl SubAssign for P{
    fn sub_assign(&mut self,a:P){
        *self=*self-a;
    }
}
impl MulAssign<i64> for P{
    fn mul_assign(&mut self,a:i64){
        *self=*self*a;
    }
}
impl DivAssign<i64> for P{
    fn div_assign(&mut self,a:i64){
        *self=*self/a;
    }
}
impl<T:Index<usize>> Index<P> for Vec<T>{
    type Output=T::Output;
    fn index(&self,idx:P)->&T::Output{
        &self[idx.0 as usize][idx.1 as usize]
    }
}
impl<T:IndexMut<usize>> IndexMut<P> for Vec<T>{
    fn index_mut(&mut self,idx:P)->&mut T::Output{
        &mut self[idx.0 as usize][idx.1 as usize]
    }
}


#[allow(unused)]
const DD:[P;4]=[P(0,-1),P(-1,0),P(0,1),P(1,0)];
#[allow(unused)]
const DX:[P;8]=[P(0,-1),P(-1,-1),P(-1,0),P(-1,1),P(0,1),P(1,1),P(1,0),P(1,-1)];


#[derive(PartialEq,PartialOrd)]
struct O<T:PartialEq+PartialOrd>(T);
impl<T:PartialEq+PartialOrd> Eq for O<T>{} 
impl<T:PartialEq+PartialOrd> Ord for O<T>{
    fn cmp(&self,a:&O<T>)->std::cmp::Ordering{
        self.0.partial_cmp(&a.0).unwrap()
    }
}





// https://github.com/terry-u16/ahc018
// ありがとうございます。


use std::collections::HashMap;

macro_rules! dict {
    ($($key:expr => $value:expr,)+) => { dict!($($key => $value),+) };
    ($($key:expr => $value:expr),*) => {
        {
            let mut dict = HashMap::new();
            $(dict.insert($key, to_f32($value));)*
            dict
        }
    };
}

#[derive(Debug, Clone)]
pub struct UNetWeightDict {
    dict: HashMap<&'static str, Vec<f32>>,
}

impl UNetWeightDict {
    pub fn new(N:usize, xy2_only:bool) -> Self {
        // trained weight
        let dict=if N<=24{
            if xy2_only{
                include!("data/weight24_xy2.txt")
            } else {
                include!("data/weight24.txt")
            }
        } else if N<=32 {
            if xy2_only{
                include!("data/weight32_xy2.txt")
            } else {
                include!("data/weight32.txt")
            }
        } else if N<=40 {
            if xy2_only{
                include!("data/weight40_xy2.txt")
            } else {
                include!("data/weight40.txt")
            }
        } else if N<=48 {
            if xy2_only{
                include!("data/weight48_xy2.txt")
            } else {
                include!("data/weight48.txt")
            }
        } else if N<=56 {
            if xy2_only{
                include!("data/weight56_xy2.txt")
            } else {
                include!("data/weight56.txt")
            }
        } else if N<=64 {
            if xy2_only{
                include!("data/weight64_xy2.txt")
            } else {
                include!("data/weight64.txt")
            }
        } else {
            panic!();
        };
        Self { dict }
    }

    pub fn get(&self, key: &str) -> Vec<f32> {
        self.dict[key].clone()
    }
}


#[allow(dead_code)]
fn to_f32(data: &[u8]) -> Vec<f32> {
    const BASE64_MAP: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut stream = vec![];

    let mut cursor = 0;

    while cursor + 4 <= data.len() {
        let mut buffer = 0u32;

        for i in 0..4 {
            let c = data[cursor + i];
            let shift = 6 * (3 - i);

            for (i, &d) in BASE64_MAP.iter().enumerate() {
                if c == d {
                    buffer |= (i as u32) << shift;
                }
            }
        }

        for i in 0..3 {
            let shift = 8 * (2 - i);
            let value = (buffer >> shift) as u8;
            stream.push(value);
        }

        cursor += 4;
    }

    let mut result = vec![];
    cursor = 0;

    while cursor + 4 <= stream.len() {
        let p = stream.as_ptr() as *const f32;
        let x = unsafe { *p.offset(cursor as isize / 4) };
        result.push(x);
        cursor += 4;
    }

    result
}





use itertools::{izip, Itertools};
use ndarray::{s, stack, Array, Array1, Array2, Array3, Array4, Axis};

pub trait NNModule {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32>;
}

#[derive(Debug, Clone)]
pub struct UNet {
    double_conv1: DoubleConvBlock,
    double_conv2: DoubleConvBlock,
    double_conv3: DoubleConvBlock,
    double_conv4: DoubleConvBlock,
    double_conv5: DoubleConvBlock,
    double_conv6: DoubleConvBlock,
    double_conv7: DoubleConvBlock,
    up_conv1: UpConvBlock,
    up_conv2: UpConvBlock,
    up_conv3: UpConvBlock,
    last_conv: Conv2d,
    maxpool: MaxPoolX2,
    sigmoid: Sigmoid,
}

const C:usize=16;

impl UNet {
    pub fn new(N:usize, xy2_only:bool) -> Self {
        let weight = UNetWeightDict::new(N, xy2_only);
        Self {
            double_conv1: DoubleConvBlock::new_dict(2, C, 1, &weight),
            double_conv2: DoubleConvBlock::new_dict(C, C*2, 2, &weight),
            double_conv3: DoubleConvBlock::new_dict(C*2, C*4, 3, &weight),
            double_conv4: DoubleConvBlock::new_dict(C*4, C*8, 4, &weight),
            double_conv5: DoubleConvBlock::new_dict(C*8, C*4, 5, &weight),
            double_conv6: DoubleConvBlock::new_dict(C*4, C*2, 6, &weight),
            double_conv7: DoubleConvBlock::new_dict(C*2, C, 7, &weight),
            up_conv1: UpConvBlock::new_dict(C*8, C*4, 1, &weight),
            up_conv2: UpConvBlock::new_dict(C*4, C*2, 2, &weight),
            up_conv3: UpConvBlock::new_dict(C*2, C, 3, &weight),
            last_conv: Conv2d::from_vec(C, 1, weight.get("conv1.weight"), weight.get("conv1.bias")),
            maxpool: MaxPoolX2,
            sigmoid: Sigmoid,
        }
    }
}

impl NNModule for UNet {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        // encoder
        let x = self.double_conv1.apply(x);
        let x1 = x.clone();
        let x = self.maxpool.apply(&x);

        let x = self.double_conv2.apply(&x);
        let x2 = x.clone();
        let x = self.maxpool.apply(&x);

        let x = self.double_conv3.apply(&x);
        let x3 = x.clone();
        let x = self.maxpool.apply(&x);

        // middle
        let x = self.double_conv4.apply(&x);

        // decoder
        let x = self.up_conv1.apply(&x);
        let x = stack![Axis(0), x3, x];
        let x = self.double_conv5.apply(&x);

        let x = self.up_conv2.apply(&x);
        let x = stack![Axis(0), x2, x];
        let x = self.double_conv6.apply(&x);

        let x = self.up_conv3.apply(&x);
        let x = stack![Axis(0), x1, x];
        let x = self.double_conv7.apply(&x);

        let x = self.last_conv.apply(&x);
        let x = self.sigmoid.apply(&x);

        x
    }
}

#[derive(Debug, Clone)]
struct DoubleConvBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    batch_norm1: BatchNorm2d,
    batch_norm2: BatchNorm2d,
    relu: Relu,
}

impl DoubleConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        conv1_weight: Vec<f32>,
        conv1_bias: Vec<f32>,
        conv2_weight: Vec<f32>,
        conv2_bias: Vec<f32>,
        bn1_weight: Vec<f32>,
        bn1_bias: Vec<f32>,
        bn1_mean: Vec<f32>,
        bn1_var: Vec<f32>,
        bn2_weight: Vec<f32>,
        bn2_bias: Vec<f32>,
        bn2_mean: Vec<f32>,
        bn2_var: Vec<f32>,
    ) -> Self {
        let conv1 = Conv2d::from_vec(in_channels, out_channels, conv1_weight, conv1_bias);
        let batch_norm1 =
            BatchNorm2d::from_vec(out_channels, bn1_weight, bn1_bias, bn1_mean, bn1_var);
        let conv2 = Conv2d::from_vec(out_channels, out_channels, conv2_weight, conv2_bias);
        let batch_norm2 =
            BatchNorm2d::from_vec(out_channels, bn2_weight, bn2_bias, bn2_mean, bn2_var);
        let relu = Relu;

        Self {
            conv1,
            conv2,
            batch_norm1,
            batch_norm2,
            relu,
        }
    }

    fn new_dict(in_channels: usize, out_channels: usize, id: usize, dict: &UNetWeightDict) -> Self {
        Self::new(
            in_channels,
            out_channels,
            dict.get(&format!("TCB{}.conv1.weight", id)),
            dict.get(&format!("TCB{}.conv1.bias", id)),
            dict.get(&format!("TCB{}.conv2.weight", id)),
            dict.get(&format!("TCB{}.conv2.bias", id)),
            dict.get(&format!("TCB{}.bn1.weight", id)),
            dict.get(&format!("TCB{}.bn1.bias", id)),
            dict.get(&format!("TCB{}.bn1.running_mean", id)),
            dict.get(&format!("TCB{}.bn1.running_var", id)),
            dict.get(&format!("TCB{}.bn2.weight", id)),
            dict.get(&format!("TCB{}.bn2.bias", id)),
            dict.get(&format!("TCB{}.bn2.running_mean", id)),
            dict.get(&format!("TCB{}.bn2.running_var", id)),
        )
    }
}

impl NNModule for DoubleConvBlock {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x = self.conv1.apply(x);
        let x = self.batch_norm1.apply(&x);
        let x = self.relu.apply(&x);
        let x = self.conv2.apply(&x);
        let x = self.batch_norm2.apply(&x);
        let x = self.relu.apply(&x);
        x
    }
}

#[derive(Debug, Clone)]
struct UpConvBlock {
    upsample: BilinearX2,
    conv: Conv2d,
    batch_norm1: BatchNorm2d,
    batch_norm2: BatchNorm2d,
    relu: Relu,
}

impl UpConvBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        conv_weight: Vec<f32>,
        conv_bias: Vec<f32>,
        bn1_weight: Vec<f32>,
        bn1_bias: Vec<f32>,
        bn1_mean: Vec<f32>,
        bn1_var: Vec<f32>,
        bn2_weight: Vec<f32>,
        bn2_bias: Vec<f32>,
        bn2_mean: Vec<f32>,
        bn2_var: Vec<f32>,
    ) -> Self {
        let upsample = BilinearX2;
        let batch_norm1 =
            BatchNorm2d::from_vec(in_channels, bn1_weight, bn1_bias, bn1_mean, bn1_var);
        let conv = Conv2d::from_vec(in_channels, out_channels, conv_weight, conv_bias);
        let batch_norm2 =
            BatchNorm2d::from_vec(out_channels, bn2_weight, bn2_bias, bn2_mean, bn2_var);
        let relu = Relu;

        Self {
            upsample,
            conv,
            batch_norm1,
            batch_norm2,
            relu,
        }
    }

    fn new_dict(in_channels: usize, out_channels: usize, id: usize, dict: &UNetWeightDict) -> Self {
        Self::new(
            in_channels,
            out_channels,
            dict.get(&format!("UC{}.conv.weight", id)),
            dict.get(&format!("UC{}.conv.bias", id)),
            dict.get(&format!("UC{}.bn1.weight", id)),
            dict.get(&format!("UC{}.bn1.bias", id)),
            dict.get(&format!("UC{}.bn1.running_mean", id)),
            dict.get(&format!("UC{}.bn1.running_var", id)),
            dict.get(&format!("UC{}.bn2.weight", id)),
            dict.get(&format!("UC{}.bn2.bias", id)),
            dict.get(&format!("UC{}.bn2.running_mean", id)),
            dict.get(&format!("UC{}.bn2.running_var", id)),
        )
    }
}

impl NNModule for UpConvBlock {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x = self.upsample.apply(x);
        let x = self.batch_norm1.apply(&x);
        let x = self.conv.apply(&x);
        let x = self.batch_norm2.apply(&x);
        let x = self.relu.apply(&x);
        x
    }
}

#[derive(Debug, Clone)]
struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    /// [C_out, C_in, K, K]
    weights: Array4<f32>,
    bias: Array1<f32>,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        weights: Array4<f32>,
        bias: Array1<f32>,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            weights,
            bias,
        }
    }

    fn from_vec(c_in: usize, c_out: usize, weight: Vec<f32>, bias: Vec<f32>) -> Self {
        const KERNEL_SIZE: usize = 3;
        let w = Array4::from_shape_vec([c_out, c_in, KERNEL_SIZE, KERNEL_SIZE], weight).unwrap();
        let b = Array1::from_shape_vec([c_out], bias).unwrap();
        Self::new(c_in, c_out, w, b)
    }
}

impl NNModule for Conv2d {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        assert_eq!(x.len_of(Axis(0)), self.in_channels);
        let x_shape = x.shape();
        let mut y = Array::zeros((self.out_channels, x_shape[1], x_shape[2]));
        let pads = x
            .outer_iter()
            .map(|x| {
                let x_shape = x.shape();
                let dim0 = x_shape[0];
                let dim1 = x_shape[1];

                // padding
                let mut pad: Array2<f32> = Array::zeros((dim0 + 2, dim1 + 2));
                pad.slice_mut(s![1..=dim0, 1..=dim1]).assign(&x);
                pad
            })
            .collect_vec();

        for (mut y, kernel) in y.outer_iter_mut().zip(self.weights.outer_iter()) {
            for (pad, kernel) in pads.iter().zip(kernel.outer_iter()) {
                for (row, mut y) in y.outer_iter_mut().enumerate() {
                    for (col, y) in y.iter_mut().enumerate() {
                        *y += pad[[row, col]] * kernel[[0, 0]];
                        *y += pad[[row, col + 1]] * kernel[[0, 1]];
                        *y += pad[[row, col + 2]] * kernel[[0, 2]];
                        *y += pad[[row + 1, col]] * kernel[[1, 0]];
                        *y += pad[[row + 1, col + 1]] * kernel[[1, 1]];
                        *y += pad[[row + 1, col + 2]] * kernel[[1, 2]];
                        *y += pad[[row + 2, col]] * kernel[[2, 0]];
                        *y += pad[[row + 2, col + 1]] * kernel[[2, 1]];
                        *y += pad[[row + 2, col + 2]] * kernel[[2, 2]];
                    }
                }
            }
        }

        for (mut y, bias) in y.outer_iter_mut().zip(self.bias.iter()) {
            for y in y.iter_mut() {
                *y += *bias;
            }
        }

        y
    }
}

#[derive(Debug, Clone, Copy)]
struct Relu;

impl NNModule for Relu {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let y = x.map(|x| x.max(0.0));
        y
    }
}

#[derive(Debug, Clone, Copy)]
struct Sigmoid;

impl NNModule for Sigmoid {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let y = x.map(|x| 1.0 / (1.0 + (-x).exp()));
        y
    }
}

#[derive(Debug, Clone)]
struct BatchNorm2d {
    weights: Array1<f32>,
    bias: Array1<f32>,
    running_mean: Array1<f32>,
    running_var: Array1<f32>,
    eps: f32,
}

impl BatchNorm2d {
    fn new(
        weights: Array1<f32>,
        bias: Array1<f32>,
        running_mean: Array1<f32>,
        running_var: Array1<f32>,
        eps: f32,
    ) -> Self {
        Self {
            weights,
            bias,
            running_mean,
            running_var,
            eps,
        }
    }

    fn from_vec(
        channel: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
    ) -> Self {
        const EPS: f32 = 1e-5;
        let w = Array1::from_shape_vec([channel], weights).unwrap();
        let b = Array1::from_shape_vec([channel], bias).unwrap();
        let m = Array1::from_shape_vec([channel], mean).unwrap();
        let v = Array1::from_shape_vec([channel], var).unwrap();
        Self::new(w, b, m, v, EPS)
    }
}

impl NNModule for BatchNorm2d {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x_shape = x.shape();
        let mut y = Array3::zeros((x_shape[0], x_shape[1], x_shape[2]));

        for (x, mut y, weight, bias, mean, var) in izip!(
            x.outer_iter(),
            y.outer_iter_mut(),
            self.weights.iter(),
            self.bias.iter(),
            self.running_mean.iter(),
            self.running_var.iter()
        ) {
            let denominator_inv = 1.0 / (*var + self.eps).sqrt();
            let x = x.map(|v| (*v - mean) * denominator_inv * weight + bias);
            y.assign(&x);
        }

        y
    }
}

/// bilinear補間を用いて画像サイズを2倍に変更する
#[derive(Debug, Clone)]
struct BilinearX2;

impl NNModule for BilinearX2 {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        const EPS: f32 = 1e-5;
        let x_shape = x.shape();
        let mut y = Array3::zeros((x_shape[0], x_shape[1] * 2, x_shape[2] * 2));

        for (x, mut y) in izip!(x.outer_iter(), y.outer_iter_mut()) {
            let y_shape = y.shape();
            let y_shape_1 = y_shape[0];
            let y_shape_2 = y_shape[1];

            for row in 0..y_shape_1 {
                let pos_y = row as f32 * (x_shape[1] - 1) as f32 / (y_shape_1 - 1) as f32;
                let floor_y = (pos_y + EPS).floor() as usize;
                let ceil_y = (pos_y - EPS).ceil() as usize;
                let dy = pos_y - floor_y as f32;

                for col in 0..y_shape_2 {
                    let pos_x = col as f32 * (x_shape[2] - 1) as f32 / (y_shape_2 - 1) as f32;
                    let floor_x = (pos_x + EPS).floor() as usize;
                    let ceil_x = (pos_x - EPS).ceil() as usize;
                    let dx = pos_x - floor_x as f32;

                    let x00 = x[[floor_y, floor_x]];
                    let x01 = x[[floor_y, ceil_x]];
                    let x10 = x[[ceil_y, floor_x]];
                    let x11 = x[[ceil_y, ceil_x]];

                    let x0 = x00 * (1.0 - dx) + x01 * dx;
                    let x1 = x10 * (1.0 - dx) + x11 * dx;
                    y[[row, col]] = x0 * (1.0 - dy) + x1 * dy;
                }
            }
        }

        y
    }
}

/// 2x2のkernelでstride=2のMaxPoolingを行う
#[derive(Debug, Clone, Copy)]
struct MaxPoolX2;

impl NNModule for MaxPoolX2 {
    fn apply(&self, x: &Array3<f32>) -> Array3<f32> {
        let x_shape = x.shape();
        assert!(x_shape[1] % 2 == 0);
        assert!(x_shape[2] % 2 == 0);
        let mut y = Array3::zeros((x_shape[0], x_shape[1] / 2, x_shape[2] / 2));

        for (x, mut y) in izip!(x.outer_iter(), y.outer_iter_mut()) {
            let y_shape = y.shape();
            let y_shape_1 = y_shape[0];
            let y_shape_2 = y_shape[1];

            for row in 0..y_shape_1 {
                for col in 0..y_shape_2 {
                    let x00 = x[[row * 2, col * 2]];
                    let x01 = x[[row * 2, col * 2 + 1]];
                    let x10 = x[[row * 2 + 1, col * 2]];
                    let x11 = x[[row * 2 + 1, col * 2 + 1]];

                    let max = x00.max(x01).max(x10).max(x11);
                    y[[row, col]] = max;
                }
            }
        }

        y
    }
}





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

    pub fn rangei(a:i64,b:i64)->i64{
        (next()%((b-a) as usize)) as i64+a
    }

    pub fn shuffle<T>(list:&mut [T]){
        for i in (0..list.len()).rev(){
            list.swap(i,next()%(i+1));
        }
    }
}