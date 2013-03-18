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

#ifndef __itkMaskedOtsuThresholdImageCalculator_txx
#define __itkMaskedOtsuThresholdImageCalculator_txx

#include "itkMaskedOtsuThresholdImageCalculator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkMinimumMaximumImageCalculator.h"

#include "vnl/vnl_math.h"

namespace itk
{ 
    
/**
 * Constructor
 */
template<class TInputImage, class TMaskImage >
MaskedOtsuThresholdImageCalculator<TInputImage,TMaskImage>
::MaskedOtsuThresholdImageCalculator()
{
  m_Image = NULL;
  m_Threshold = NumericTraits<PixelType>::Zero;
  m_NumberOfHistogramBins = 128;
  m_RegionSetByUser = false;
}


/*
 * Compute the Otsu's threshold
 */
template<class TInputImage, class TMaskImage >
void
MaskedOtsuThresholdImageCalculator<TInputImage,TMaskImage>
::Compute( typename TMaskImage::ConstPointer pMask )
{

  unsigned int j;

  if ( !m_Image ) { return; }
  if( !m_RegionSetByUser )
    {
    m_Region = m_Image->GetRequestedRegion();
    }

  double totalPixels = (double) m_Region.GetNumberOfPixels();
  if ( totalPixels == 0 ) { return; }


  // compute image max and min, this needs to be done over the mask

  typedef ImageRegionConstIteratorWithIndex<TInputImage> Iterator;

  // iterate over the mask
  
  Iterator iter( m_Image, m_Region );
  Iterator iterMask( pMask, m_Region );
  
  iter.GoToBegin();
  iterMask.GoToBegin();
  
  PixelType imageMin = 0;
  PixelType imageMax = 0;

  unsigned int uiNumberOfConsideredVoxels = 0;
  
  while ( !iter.IsAtEnd() && !iterMask.IsAtEnd() )
    {
    PixelType value = iter.Get();
    
    if ( iterMask.Get()>0 ) 
      {
      if ( uiNumberOfConsideredVoxels==0 )
        {
        imageMin = value;
        imageMax = value;
        }
      else
        {
        // already have a valid value

        if ( value<imageMin ) imageMin = value;
        if ( value>imageMax ) imageMax = value;

        }
      uiNumberOfConsideredVoxels++;
      }

    ++iter;
    ++iterMask;

    }

  if ( imageMin >= imageMax )
    {
    m_Threshold = imageMin;
    return;
    }

  std::cout << "imageMin = " << imageMin << std::endl;
  std::cout << "imageMax = " << imageMax << std::endl;


  // create a histogram
  std::vector<double> relativeFrequency;
  relativeFrequency.resize( m_NumberOfHistogramBins );
  for ( j = 0; j < m_NumberOfHistogramBins; j++ )
    {
    relativeFrequency[j] = 0.0;
    }

  double binMultiplier = (double) m_NumberOfHistogramBins /
    (double) ( imageMax - imageMin );

  // iterate over the mask
  
  iter.GoToBegin();
  iterMask.GoToBegin();

  while ( !iter.IsAtEnd() && !iterMask.IsAtEnd() )
    {
    unsigned int binNumber;
    PixelType value = iter.Get();
    
    if ( iterMask.Get()>0 ) 
      {
      // Is in mask, so use this value

      if ( value == imageMin ) 
        {
        binNumber = 0;
        }
      else
        {
        binNumber = (unsigned int) vcl_ceil((value - imageMin) * binMultiplier ) - 1;
        if ( binNumber == m_NumberOfHistogramBins ) // in case of rounding errors
          {
          binNumber -= 1;
          }
        }
      
      relativeFrequency[binNumber] += 1.0;
      }
    ++iter;
    ++iterMask;
    
    }
  
  std::cout << "Number of considered voxels = " << uiNumberOfConsideredVoxels << std::endl;

  // normalize the frequencies
  double totalMean = 0.0;
  for ( j = 0; j < m_NumberOfHistogramBins; j++ )
    {
    relativeFrequency[j] /= uiNumberOfConsideredVoxels;
    totalMean += (j+1) * relativeFrequency[j];
    }


  /*for ( j = 0; j < m_NumberOfHistogramBins; j++ )
    {
    std::cout << "RF[" << j << "] = " << relativeFrequency[j] << std::endl;
    }*/

  // compute Otsu's threshold by maximizing the between-class
  // variance
  double freqLeft = relativeFrequency[0];
  double meanLeft = 1.0;
  double meanRight = ( totalMean - freqLeft ) / ( 1.0 - freqLeft );

  double maxVarBetween = freqLeft * ( 1.0 - freqLeft ) *
    vnl_math_sqr( meanLeft - meanRight );
  int maxBinNumber = 0;

  double freqLeftOld = freqLeft;
  double meanLeftOld = meanLeft;

  for ( j = 1; j < m_NumberOfHistogramBins; j++ )
    {
    freqLeft += relativeFrequency[j];
    meanLeft = ( meanLeftOld * freqLeftOld + 
                 (j+1) * relativeFrequency[j] ) / freqLeft;
    if (freqLeft == 1.0)
      {
      meanRight = 0.0;
      }
    else
      {
      meanRight = ( totalMean - meanLeft * freqLeft ) / 
        ( 1.0 - freqLeft );
      }
    double varBetween = freqLeft * ( 1.0 - freqLeft ) *
      vnl_math_sqr( meanLeft - meanRight );
   
    if ( varBetween > maxVarBetween )
      {
      maxVarBetween = varBetween;
      maxBinNumber = j;
      }

    // cache old values
    freqLeftOld = freqLeft;
    meanLeftOld = meanLeft; 

    } 

  m_Threshold = static_cast<PixelType>( imageMin + 
                                        ( maxBinNumber + 1 ) / binMultiplier );

}

template<class TInputImage, class TMaskImage>
void
MaskedOtsuThresholdImageCalculator<TInputImage,TMaskImage>
::SetRegion( const RegionType & region )
{
  m_Region = region;
  m_RegionSetByUser = true;
}

  
template<class TInputImage, class TMaskImage>
void
MaskedOtsuThresholdImageCalculator<TInputImage,TMaskImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Threshold: " << m_Threshold << std::endl;
  os << indent << "NumberOfHistogramBins: " << m_NumberOfHistogramBins << std::endl;
  os << indent << "Image: " << m_Image.GetPointer() << std::endl;
}

} // end namespace itk

#endif
