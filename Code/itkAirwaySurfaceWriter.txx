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

#ifndef __itkAirwaySurfaceWriter_txx
#define __itkAirwaySurfaceWriter_txx

#include "itkAirwaySurfaceWriter.h"

#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkFastMarchingImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageToVTKImageFilter.h>
#include <itkVTKImageToImageFilter.h>
#include <vtkContourFilter.h>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

namespace itk {

template<class TInputImage, class TMaskImage>
AirwaySurfaceWriter<TInputImage, TMaskImage>
::AirwaySurfaceWriter()
{
  m_pMaskImage     = 0;
  m_FileName = "";
  m_UseFastMarching = false;
}

template<class TInputImage, class TMaskImage>
void
AirwaySurfaceWriter<TInputImage, TMaskImage>
::GenerateData()
{
  typedef itk::Image<float, 3> TFloatImage;
  typedef itk::MaskImageFilter< TInputImage, TFloatImage, TInputImage > MaskFilterType;
  typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();

  //
  //Dilatation
  //

  //Watchout, we need the airway to be 1 and the outside 0

  if (m_UseFastMarching)
    {
    //Fast marching for dilatation on the segemented image
    //Necessaries typedefs
    typedef itk::FastMarchingImageFilter< TFloatImage, TFloatImage >  FastMarchingFilterType;
    typedef FastMarchingFilterType::NodeContainer  NodeContainer; 
    typedef FastMarchingFilterType::NodeType NodeType; 
    typedef itk::ImageRegionConstIterator< TMaskImage > ConstIteratorType; 
    
    //Instantiations
    typename FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();
    typename NodeContainer::Pointer seeds = NodeContainer::New();
    seeds->Initialize();

    //Nodes are created as stack variables and 
    //initialized with a value and an itk::Index position. NodeType node;
    NodeType node;

    //WATCHOUT ! Here we want to expand the trachea !!!!!
    node.SetValue( 0.0 ); 
    
    // loop through the output image 
    // and set all voxels to 0 seed voxels
    ConstIteratorType it( this->m_pMaskImage,
                          this->m_pMaskImage->GetLargestPossibleRegion() );


    unsigned int uiNumberOfSeeds = 0;
    for ( it.GoToBegin(); !it.IsAtEnd(); ++it ) 
      {
      if ( it.Get() > 0 )
        {
        node.SetIndex( it.GetIndex() );
        seeds->InsertElement( uiNumberOfSeeds++, node );
        }
      }
    //std::cout<<uiNumberOfSeeds<<std::endl;

    //The set of seed nodes is now passed to the
    // FastMarchingImageFilter with the method SetTrialPoints().
    fastMarching->SetTrialPoints( seeds );

    // The FastMarchingImageFilter requires the user to specify 
    //the size of the image to be produced as output.
    //This is done using the SetOutputSize(). 

    fastMarching->SetInput( NULL );
    fastMarching->SetSpeedConstant( 1.0 );  // to solve a simple Eikonal equation

    fastMarching->SetOutputSize( this->m_pMaskImage->GetBufferedRegion().GetSize() );
    fastMarching->SetOutputRegion( this->m_pMaskImage->GetBufferedRegion() );
    fastMarching->SetOutputSpacing( this->m_pMaskImage->GetSpacing() );
    fastMarching->SetOutputOrigin( this->m_pMaskImage->GetOrigin() );
    fastMarching->SetStoppingValue( 1.0 );

    fastMarching->Update();

    //Invert the mask, so we get the inside of it, not the outside
    typedef itk::BinaryThresholdImageFilter< TFloatImage, TFloatImage >
      ThresholdingFilterType;
    typename ThresholdingFilterType::Pointer thresholdDilation = ThresholdingFilterType::New();

    thresholdDilation->SetLowerThreshold( 0.0 ); 
    thresholdDilation->SetUpperThreshold( 1.0 );
    thresholdDilation->SetOutsideValue( 0.0 ); 
    thresholdDilation->SetInsideValue( 1.0 );
    thresholdDilation->SetInput( fastMarching->GetOutput() );
    thresholdDilation->Update();
  
    maskFilter->SetMaskImage(thresholdDilation->GetOutput());
    }
  else
    {
    typedef itk::BinaryBallStructuringElement<float, 3> 
      TStructuringElement;
    TStructuringElement structuringElement;
    structuringElement.SetRadius(1);
    structuringElement.CreateStructuringElement();

    typedef itk::BinaryDilateImageFilter <TInputImage,
                                          TFloatImage,
                                          TStructuringElement>
      BinaryDilateImageFilterType;
    typename BinaryDilateImageFilterType::Pointer dilateFilter
      = BinaryDilateImageFilterType::New();
    dilateFilter->SetDilateValue(1);
    dilateFilter->SetInput(this->m_pMaskImage);
    dilateFilter->SetKernel(structuringElement);
    dilateFilter->Update();

    maskFilter->SetMaskImage(dilateFilter->GetOutput());
    }

  //
  //Masking
  //
  
  maskFilter->SetInput(this->GetInput());
  maskFilter->SetOutsideValue(std::numeric_limits<InputPixelType>::max());
  maskFilter->Update();

  //
  //Threshold
  //

  typedef itk::OtsuThresholdImageFilter<TInputImage, TInputImage>  otsuThresholdFilterType;
  typename otsuThresholdFilterType::Pointer otsuThreshold = otsuThresholdFilterType::New();
  otsuThreshold->SetInsideValue( 0 );
  otsuThreshold->SetOutsideValue( 1 );
  otsuThreshold->SetInput( maskFilter->GetOutput() );
  otsuThreshold->Update();

  //
  //Contour
  //
  
  //Conversion to VTK
  typedef itk::ImageToVTKImageFilter<TInputImage> adaptatorFromITKtoVTKType;
  typename adaptatorFromITKtoVTKType::Pointer toVTKFilter = adaptatorFromITKtoVTKType::New();
  toVTKFilter->SetInput( otsuThreshold->GetInput() );
  toVTKFilter->Update();

  //Apply VTK Filter
  vtkSmartPointer<vtkContourFilter> contourFilter = vtkContourFilter::New();
  contourFilter->SetInput(toVTKFilter->GetOutput());
  contourFilter->SetValue(0, 1.0);
  contourFilter->Update();
  
  //Transform it to IJK (i.e. pixel space) 
  // <-> equivalent to a rotation of 180 degrees around the Z axis
  vtkSmartPointer<vtkTransform> transform = vtkTransform::New();
  transform->RotateZ(180.0);
  
  vtkSmartPointer<vtkTransformPolyDataFilter> transformer = 
    vtkTransformPolyDataFilter::New();
  transformer->SetInput(contourFilter->GetOutput());
  transformer->SetTransform(transform);
  transformer->Update();
  
  //Write to file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkXMLPolyDataWriter::New();
  writer->SetFileName(this->m_FileName.c_str());

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(transformer->GetOutput());
#else
  writer->SetInputData(transformer->GetOutput());
#endif

  writer->Write();
}

template<class TInputImage, class TMaskImage>
void 
AirwaySurfaceWriter<TInputImage, TMaskImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  std::cout<<indent<<"Use Fast Marching: "
    <<this->m_UseFastMarching<<std::endl;
  std::cout<<indent<<"FileName: "
    <<this->m_FileName<<std::endl;
  std::cout<<indent<<"Mask Image: "<<std::endl;
  this->m_pMaskImage->Print(os);
}


}// end namespace itk
#endif
