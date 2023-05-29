/**
 * Implementation of the synchronization method described in:
 *
 * "Multiview Registration via Graph Diffusion of Dual Quaternions"
 * A.Torsello, E.Rodola, and A.Albarelli
 * Proc. CVPR 2011
 *
 * Emanuele Rodola <emanuele.rodola@usi.ch>
 * USI Lugano, July 2016
 */
 
#include "math3d.h"

using math3d::point3d;
using math3d::matrix;
using math3d::quaternion;


template<typename T> inline int sign(T v) { return (v<0)?-1:1; }

void set_quaternion_matrix(matrix<double>&M, const quaternion<double>& q, int i=0, int j=0, double w=1.0)
{
   //{{a, -b, -c, -d}, {b, a, -d, c}, {c, d, a, -b}, {d, -c, b, a}}
   M(i,j)  = w*q.w; M(i,j+1)  =-w*q.i; M(i,j+2)  =-w*q.j; M(i,j+3)  =-w*q.k;
   M(i+1,j)= w*q.i; M(i+1,j+1)= w*q.w; M(i+1,j+2)=-w*q.k; M(i+1,j+3)= w*q.j;
   M(i+2,j)= w*q.j; M(i+2,j+1)= w*q.k; M(i+2,j+2)= w*q.w; M(i+2,j+3)=-w*q.i;
   M(i+3,j)= w*q.k; M(i+3,j+1)=-w*q.j; M(i+3,j+2)= w*q.i; M(i+3,j+3)= w*q.w;
}

struct dual_quaternion
{
   quaternion<double> R, tR_2;

   dual_quaternion(double v=1.0) : R(v), tR_2(0) {}

   static const double dq_epsilon= 1e-8;

   static dual_quaternion rigid_transformation(const quaternion<double>& r, const point3d& t)
   {
      dual_quaternion result;
      result.R = r;
      result.tR_2 = (quaternion<double>::convert(t) * r) *= 0.5;
      return result;
   }
   static dual_quaternion convert(const double* p)
   {
      dual_quaternion result;
      result.R = quaternion<double>::convert(p);
      result.tR_2 = quaternion<double>::convert(p+4);
      return result;
   }

   dual_quaternion& normalize()
   {
      double n=norm(R)*sign(R.w);
      R*=1.0/n;
      tR_2*=1.0/n;
      double d=dot(R,tR_2);
      //tR_2 += (-d)*R;
      quaternion<double> r2=R;
      r2 *= -d;
      tR_2 += r2;
      return *this;
   }

   point3d get_translation()
   {
      quaternion<double> t = tR_2 * ~R;
      point3d result;
      result.x = 2*t.i;
      result.y = 2*t.j;
      result.z = 2*t.k;
      return result;
   }

   void to_vector(double* p)
   { R.to_vector(p); tR_2.to_vector(p+4); }

   dual_quaternion& operator += (const dual_quaternion& a)
                                { R+=a.R; tR_2+=a.tR_2; return *this; }

   dual_quaternion& operator *= (double a)
                                {R*=a; tR_2*=a; return *this;}

   dual_quaternion& log()	//computes log map tangent at identity
   {				//assumes qual_quaternion is unitary
      const double h0=std::acos(R.w);
      if ( h0*h0 < dq_epsilon ) //small angle approximation: sin(h0)=h0, cos(h0)=1
      {
         R.w=0.0;
         R *= 0.5;
         tR_2.w=0.0;
         tR_2 *= 0.5;
      }
      else
      {
         R.w=0.0;
         const double ish0=1.0/norm(R);
         //R *= ish0;
         math3d::normalize(R); //R=s0
         const double he=-tR_2.w*ish0;
         tR_2.w=0.0;

         quaternion<double> Rp(R);
         Rp *= -dot(R,tR_2)/dot(R,R);
         tR_2 += Rp;
         tR_2 *= ish0; //tR_2=se

         tR_2 *= h0;
         Rp=R;
         Rp*=he;
         tR_2 += Rp;
         tR_2 *= 0.5;
         R *= h0*0.5;
      }

      return *this;
   }

   dual_quaternion& exp()	//computes exp map tangent at identity
   {				//assumes qual_quaternion is on tangent space
      const double h0=2.0*norm(R);

      if (h0*h0 < dq_epsilon) //small angle approximation: sin(h0)=h0, cos(h0)=1
      {
         R *= 2.0;
         R.w = 1.0;
         tR_2 *= 2.0;
         //normalize();
      }
      else
      {
         const double he= 4.0*math3d::dot(tR_2,R)/h0;
         const double sh0=sin(h0), ch0=cos(h0);
         quaternion<double> Rp(R);
         Rp *= -dot(R,tR_2)/dot(R,R);
         tR_2 += Rp;
         tR_2 *=2.0/h0; //tR_2=se


         tR_2 *= sh0;
         Rp=R;
         Rp *= he*ch0*2.0/h0;
         tR_2 += Rp;
         tR_2.w = -he*sh0;

         R *= sh0*2.0/h0;
         R.w = ch0;
      }
      normalize();
      return *this;
   }
};


dual_quaternion operator * (const dual_quaternion&a, const dual_quaternion& b)
{
   dual_quaternion result;
   result.R=a.R*b.R;
   result.tR_2=a.R*b.tR_2 + a.tR_2*b.R;
   return result;
}

dual_quaternion operator ~(const dual_quaternion& a)
{ dual_quaternion result; result.R = ~a.R; result.tR_2 = ((~a.tR_2)*=-1); return result; }

dual_quaternion operator !(const dual_quaternion& a)
{ dual_quaternion result; result.R = ~a.R; result.tR_2 = ~a.tR_2; return result; }

double dot(const dual_quaternion& a, const dual_quaternion& b)
{ return dot(a.R,b.R) + dot(a.tR_2,b.tR_2); }

void set_dual_quaternion_matrix(matrix<double>& M, const dual_quaternion& dq, int i=0, int j=0, double w=1.0)
{
   set_quaternion_matrix(M,dq.R,i,j,w);
   M(i,j+4)=M(i,j+5)=M(i,j+6)=M(i,j+7)=0;
   M(i+1,j+4)=M(i+1,j+5)=M(i+1,j+6)=M(i+1,j+7)=0;
   M(i+2,j+4)=M(i+2,j+5)=M(i+2,j+6)=M(i+2,j+7)=0;
   M(i+3,j+4)=M(i+3,j+5)=M(i+3,j+6)=M(i+3,j+7)=0;
   set_quaternion_matrix(M,dq.tR_2,i+4,j,w);set_quaternion_matrix(M,dq.R,i+4,j+4,w);
}

dual_quaternion log(dual_quaternion a) { return a.log(); }
dual_quaternion exp(dual_quaternion a) { return a.exp(); }


std::ostream& operator << (std::ostream& out, const dual_quaternion& dq)
{
   return out << "( " << dq.R.w << ", " << dq.R.i << ", " << dq.R.j << ", " << dq.R.k << ",  "
         << dq.tR_2.w << ", " << dq.tR_2.i << ", " << dq.tR_2.j << ", " << dq.tR_2.k << " )";
}


class transducer
{
public:
   int n;
   std::vector<std::vector< std::pair<int,dual_quaternion> > > DM;
   std::vector<std::vector<double> > W;
   std::vector<double> WM;
   std::vector<dual_quaternion> Q;

public:
   transducer(int N) : n(N), DM(N), W(N), WM(N,0.0), Q(N,dual_quaternion(1.0)) {}

   void add_transformation(int i, int j, const dual_quaternion& dq, double weight=1.0)
   { //add transformation that maps i coordinates in j coordinates
      DM[i].push_back(std::pair<int, dual_quaternion>(j, !dq));
      DM[j].push_back(std::pair<int, dual_quaternion>(i, dq));
      W[i].push_back(weight);
      W[j].push_back(weight);
      WM[i] += weight;
      WM[j] += weight;
   }

   void add_transformation(int i, int j, const quaternion<double>& r, const point3d& t, double weight=1.0)
   {
      add_transformation(i, j, dual_quaternion::rigid_transformation(r,t), weight);
   }

   void linear_transduce(const int diffusion_iterations=1000);
   void manifold_transduce(const int diffusion_iterations=1000, const int average_iterations=4);

   void set_position(int i, const dual_quaternion& dq) { Q[i] = dq; }

   void set_position(int i, const quaternion<double>& r, const point3d& t)
   {
      set_position(i, dual_quaternion::rigid_transformation(r,t));
   }

   void get_position(int i, dual_quaternion& dq) { dq = Q[i]; }

   void get_position(int i, quaternion<double>& r, point3d& t)
   {
      dual_quaternion dq;
      get_position(i,dq);
      r = dq.R;
      t = dq.get_translation();
   }

   void get_estimate();
   double rmste();
};


void transducer::get_estimate()
{
   std::vector<int> view_stack(1,0);
   std::vector<char> view_visited(n,0);

   view_visited[0]=1;
   //std::vector<dual_quaternion> Q(n);
   int stack_pos=0;
   while (stack_pos != (int)view_stack.size())
   {
      int cur_view=view_stack[stack_pos];
      for (std::vector< std::pair<int,dual_quaternion> > ::iterator iter=DM[cur_view].begin(); iter!=DM[cur_view].end(); ++iter)
      {
         if (!view_visited[iter->first])
         {
            view_stack.push_back(iter->first);
            view_visited[iter->first]=1;
            //std::cerr << cur_view << " -> " << iter->first << " <=> " << Q[cur_view] << " * " << iter->second << " = " << iter->second * Q[cur_view] << std::endl;
            Q[iter->first] = ( iter->second * Q[cur_view] ).normalize();
         }
      }
      ++stack_pos;
   }
   if (stack_pos!=n)
   {
      std::cerr << "view graph not connected\n";
      throw std::runtime_error("view graph not connected in void transducer::get_estimate()");
   }
}

double transducer::rmste()
{
   /*
    double ste=0.0;
    int num=0;

    for (int i=0; i!=n; ++i)
    {
        for (std::vector< std::pair<int,dual_quaternion> > ::iterator iter=DM[i].begin(); iter!=DM[i].end(); ++iter)
        {
            dual_quaternion q=Q[i], qi=((!iter->second)*Q[iter->first]).normalize();
            const double w=-sign(dot(Q[i], qi ));
            q += (w*qi);
            ste += dot(q,q);
            ++num;
        }
    }
    return std::sqrt(ste/num);
    */
   double ste=0.0;
   //double tot_w=0.0;

   for (int i=0; i!=n; ++i)
   {
      std::vector<double>::iterator witer=W[i].begin();
      for (std::vector< std::pair<int,dual_quaternion> > ::iterator iter=DM[i].begin(); iter!=DM[i].end(); ++iter, ++witer)
      {
         dual_quaternion q=((!iter->second)*Q[iter->first]*!Q[i]).normalize();
         q.log();
         ste += dot(q,q)*(*witer);
         //tot_w += *witer;
      }
   }
   return std::sqrt(ste);
}

void transducer::linear_transduce(const int diffusion_iterations)
{
   get_estimate();
   std::vector<dual_quaternion> Q2(Q);
   for(int rep=0; rep!=diffusion_iterations; ++rep)
   {
      for (int i=0; i!=n; ++i)
      {
         Q2[i]=dual_quaternion(0.0);
         std::vector<double>::iterator witer=W[i].begin();
         for (std::vector< std::pair<int,dual_quaternion> > ::iterator iter=DM[i].begin(); iter!=DM[i].end(); ++iter, ++witer)
         {
            dual_quaternion qi=(!iter->second)*Q[iter->first];
            const double w=sign(math3d::dot(Q[i].R, qi.R ))*(*witer)/WM[i];
            Q2[i] += (w*qi);
         }
         Q2[i].normalize();
      }
      for (int i=0; i!=n; ++i)
         Q[i]=Q2[i];
   }
   dual_quaternion st=!Q[0];
   for (int i=0; i!=n; ++i)
      Q[i] = (Q[i]*st).normalize();
}

void transducer::manifold_transduce(const int diffusion_iterations, const int average_iterations)
{
   get_estimate();
   std::vector<dual_quaternion> Q2(Q);
   for(int rep=0; rep!=diffusion_iterations; ++rep)
   {
      for (int i=0; i!=n; ++i)
      {
         Q2[i]=dual_quaternion(0.0);
         std::vector<double>::iterator witer=W[i].begin();
         for (std::vector< std::pair<int,dual_quaternion> > ::iterator iter=DM[i].begin(); iter!=DM[i].end(); ++iter, ++witer)
         {
            dual_quaternion qi=(!iter->second)*Q[iter->first];
            const double w=sign(math3d::dot(Q[i].R, qi.R ))*(*witer)/WM[i];
            Q2[i] += (w*qi);
         }
         Q2[i].normalize();


         for(int avg_rep=0; avg_rep!=average_iterations; ++avg_rep)
         {
            dual_quaternion log_mean(0.0);
            std::vector<double>::iterator witer=W[i].begin();
            for (std::vector< std::pair<int,dual_quaternion> > ::iterator iter=DM[i].begin(); iter!=DM[i].end(); ++iter, ++witer)
            {
               dual_quaternion qi=(!iter->second)*Q[iter->first]*!Q2[i];
               qi.normalize(); //should not be needed!
               qi *= sign(qi.R.w);
               qi.log();
               qi *= *witer;
               log_mean += qi;
            }
            log_mean *= 1.0/WM[i];
            //std::cerr << "### " << i << " - " << log_mean << std::endl;
            Q2[i] = (log_mean.exp())*Q2[i];
            Q2[i].normalize(); //should not be needed!
         }

      }

      for (int i=0; i!=n; ++i)
         Q[i]=Q2[i];
   }

   dual_quaternion st=!Q[0];
   for (int i=0; i!=n; ++i)
      Q[i] = (Q[i]*st).normalize();

}
