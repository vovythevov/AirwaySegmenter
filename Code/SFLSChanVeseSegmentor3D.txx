/*=============================================================================
//  --- Airway Segmenter ---+
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0 
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Authors: Marc Niethammer, Yi Hong, Johan Andruejol
=============================================================================*/

#ifndef SFLSChanVeseSegmentor3D_hpp_
#define SFLSChanVeseSegmentor3D_hpp_

#include "SFLSChanVeseSegmentor3D.h"

#include <algorithm>


/* ============================================================
   basicInit    */
template< typename TPixel >
void
CSFLSChanVeseSegmentor3D< TPixel >
::basicInit()
{
  SuperClassType::basicInit();
  m_bUpdateMeans = true;
}  


/* ============================================================
   computeForce    */
template< typename TPixel >
void
CSFLSChanVeseSegmentor3D< TPixel >
::computeForce()
{
  this->m_force.clear();

  if ( m_bUpdateMeans )
    {
    // only upate if desired. Otherwise this is only used to clean up segmentations based on curvature
    updateMeans();
    }

  double fmax = -1e10;

  long n = this->m_lz.size();
  double* kappaOnZeroLS = new double[ n ];
  double* cvForce = new double[ n ];
  
  long i = 0;
  
  for (typename CSFLSLayer::iterator itz = this->m_lz.begin(); itz != this->m_lz.end(); ++itz, ++i)
    {
    long ix = (*itz)[0];
    long iy = (*itz)[1];
    long iz = (*itz)[2];
    
    typename itk::Image<TPixel, 3>::IndexType idx = {{ix, iy, iz}};
    
    kappaOnZeroLS[i] = this->computeKappa(ix, iy, iz);
    
    double I = this->mp_img->GetPixel(idx);
    double a = (I - m_meanIn)*(I - m_meanIn) - (I - m_meanOut)*(I - m_meanOut);
    
    fmax = fabs(a)>fmax?fabs(a):fmax;
    
    cvForce[i] = a;
    }
  

  if ( this->m_bUseEvolutionMask )
    {

    i = 0;
    for (typename CSFLSLayer::iterator itz = this->m_lz.begin(); itz != this->m_lz.end(); ++itz, ++i)
      {

      long ix = (*itz)[0];
      long iy = (*itz)[1];
      long iz = (*itz)[2];
      
      typename itk::Image<TPixel, 3>::IndexType idx = {{ix, iy, iz}};

      if ( this->mp_evolutionMask->GetPixel( idx )==0 )
        {
        this->m_force.push_back( 0 );  // in mask, don't continue evolving here
        }
      else
        {
        this->m_force.push_back(cvForce[i]/(fmax + 1e-10) +  (this->m_curvatureWeight)*kappaOnZeroLS[i]);
        }
      
      }
    }
  else
    {
    // no restrictions, go for it
    
    for (long i = 0; i < n; ++i)
      {
      this->m_force.push_back(cvForce[i]/(fmax + 1e-10) +  (this->m_curvatureWeight)*kappaOnZeroLS[i]);
      }

    }
    
  delete[] kappaOnZeroLS;
  delete[] cvForce;
}


/* ============================================================
   doSegmenation    */
template< typename TPixel >
void
CSFLSChanVeseSegmentor3D< TPixel >
::doSegmenation()
{
  /*============================================================
   * From the initial mask, generate: 1. SFLS, 2. mp_label and
   * 3. mp_phi.      
   */
  this->initializeSFLS();

  if ( !this->mp_statisticMask )
    {
    computeMeans();    
    }
  else
    {
    computeMaskedMeans();
    }

  //douher::saveAsImage2< double >(mp_phi, "initPhi.nrrd");
  for (unsigned int it = 0; it < this->m_numIter; ++it)
    {

    std::cout << "iter = " << it << std::endl;
    
    computeForce();
    
    this->normalizeForce();
    
    this->oneStepLevelSetEvolution();

    }
}


/* ============================================================
   computeMeans    */
template< typename TPixel >
void
CSFLSChanVeseSegmentor3D< TPixel >
::computeMeans()
{
  m_areaIn = 0;
  m_areaOut = 0;

  m_meanIn = 0;
  m_meanOut = 0;

  for (long ix = 0; ix < this->m_nx; ++ix)
    {
      for (long iy = 0; iy < this->m_ny; ++iy)
        {
          for (long iz = 0; iz < this->m_nz; ++iz)
            {
              typename itk::Image<TPixel, 3>::IndexType idx = {{ix, iy, iz}};
                
              double phi = this->mp_phi->GetPixel(idx);
              double v = this->mp_img->GetPixel(idx);

              if (phi <= 0)
                {
                  ++m_areaIn;
                  m_meanIn += v;
                }
              else
                {
                  ++m_areaOut;
                  m_meanOut += v;
                }
            }
        }
    }

  m_meanIn /= (m_areaIn + vnl_math::eps);
  m_meanOut /= (m_areaOut + vnl_math::eps);
}

/* ============================================================
   computeMaskedMeans    */
template< typename TPixel >
void
CSFLSChanVeseSegmentor3D< TPixel >
::computeMaskedMeans()
{
  m_areaIn = 0;
  m_areaOut = 0;

  m_meanIn = 0;
  m_meanOut = 0;

  for (long ix = 0; ix < this->m_nx; ++ix)
    {
      for (long iy = 0; iy < this->m_ny; ++iy)
        {
          for (long iz = 0; iz < this->m_nz; ++iz)
            {
              typename itk::Image<TPixel, 3>::IndexType idx = {{ix, iy, iz}};
                
              if ( this->mp_statisticMask->GetPixel( idx ) > 0 )
                {
                double phi = this->mp_phi->GetPixel(idx);
                double v = this->mp_img->GetPixel(idx);
                
                if (phi <= 0)
                  {
                  ++m_areaIn;
                  m_meanIn += v;
                  }
                else
                  {
                  ++m_areaOut;
                  m_meanOut += v;
                  }
                }
            }
        }
    }

  m_meanIn /= (m_areaIn + vnl_math::eps);
  m_meanOut /= (m_areaOut + vnl_math::eps);
}


/* ============================================================
   updateMeans    */
template< typename TPixel >
void
CSFLSChanVeseSegmentor3D< TPixel >
::updateMeans()
{

  std::cout << "Updating means" << std::endl;

  double sumIn = m_meanIn*m_areaIn;
  double sumOut = m_meanOut*m_areaOut;

  for (typename CSFLSLayer::const_iterator it = this->m_lIn2out.begin(); it != this->m_lIn2out.end(); ++it)
    {
      long ix = (*it)[0];
      long iy = (*it)[1];
      long iz = (*it)[2];

      typename itk::Image<TPixel, 3>::IndexType idx = {{ix, iy, iz}};

      bool bEvaluate = false;

      if ( !this->mp_statisticMask )
        {
        bEvaluate = true;
        }
      else
        {
        if ( this->mp_statisticMask->GetPixel(idx) > 0 )
          {
          bEvaluate = true;
          }
        }

      if ( bEvaluate )
        {
	    
        sumIn  -= this->mp_img->GetPixel(idx); 
        --m_areaIn;
        
        sumOut += this->mp_img->GetPixel(idx); 
        ++m_areaOut;

        }

    }

  for (typename CSFLSLayer::const_iterator it = this->m_lOut2in.begin(); it != this->m_lOut2in.end(); ++it)
    {
      long ix = (*it)[0];
      long iy = (*it)[1];
      long iz = (*it)[2];

      typename itk::Image<TPixel, 3>::IndexType idx = {{ix, iy, iz}};
	 
      bool bEvaluate = false;

      if ( !this->mp_statisticMask )
        {
        bEvaluate = true;
        }
      else
        {
        if ( this->mp_statisticMask->GetPixel(idx) > 0 )
          {
          bEvaluate = true;
          }
        }
   
      if ( bEvaluate )
        {
        
        sumIn  += this->mp_img->GetPixel(idx);
        ++m_areaIn;
        
        sumOut -= this->mp_img->GetPixel(idx);
        --m_areaOut;

        }
    }


  m_meanIn = (sumIn/m_areaIn + vnl_math::eps);
  m_meanOut = (sumOut/m_areaOut + vnl_math::eps);
}




#endif
