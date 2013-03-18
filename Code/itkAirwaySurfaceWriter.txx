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
#include <itkResampleImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkIdentityTransform.h>
#include <itkImageFileWriter.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkPolyDataConnectivityFilter.h>

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
  
//    maskFilter->SetMaskImage(thresholdDilation->GetOutput());

    // upsample the binary image
    typedef itk::ResampleImageFilter<TFloatImage, TFloatImage> resampleFilterType;
    typename resampleFilterType::Pointer resampleBinaryFilter = resampleFilterType::New();
    resampleBinaryFilter->SetInput( thresholdDilation->GetOutput() );

    resampleBinaryFilter->SetOutputOrigin( thresholdDilation->GetOutput()->GetOrigin() );

    InputSizeType outputBinarySize = thresholdDilation->GetOutput()->GetLargestPossibleRegion().GetSize();
    InputSpacingType outputBinarySpacing = thresholdDilation->GetOutput()->GetSpacing();
    for( int iI = 0; iI < 3; iI++ )
    {
        outputBinarySize[iI] *= 2;
        outputBinarySpacing[iI] /= 2;
    }
    resampleBinaryFilter->SetSize( outputBinarySize );
    resampleBinaryFilter->SetOutputSpacing( outputBinarySpacing );

    typedef itk::IdentityTransform<double, 3> TransformType;
    resampleBinaryFilter->SetTransform( TransformType::New() );

    resampleBinaryFilter->Update();

    typedef itk::BinaryThresholdImageFilter< TFloatImage, TFloatImage > BinaryThresholdImageFilterType;
    BinaryThresholdImageFilterType::Pointer thresholdFilter = BinaryThresholdImageFilterType::New();
    thresholdFilter->SetInput( resampleBinaryFilter->GetOutput() );
    thresholdFilter->SetLowerThreshold(0.5);
    thresholdFilter->SetUpperThreshold(1.0);
    thresholdFilter->SetInsideValue(1);
    thresholdFilter->SetOutsideValue(0);
    thresholdFilter->Update();

    maskFilter->SetMaskImage( thresholdFilter->GetOutput() );

    /*{
	typedef itk::ImageFileWriter<TFloatImage> WriterLabelType;
    	typename WriterLabelType::Pointer writerSample = WriterLabelType::New();
    	writerSample->SetInput( resampleBinaryFilter->GetOutput() );
    	std::string filename = "/playpen/Project/airwayAtlas/AirwaySegmenterInSlicer";
    	filename += "/upsample_binary.nhdr";
    	writerSample->SetFileName( filename );

    	try
      	{
      		writerSample->Update(); 
      	}
    	catch ( itk::ExceptionObject & excep )
      	{
      	std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl;
     	}
    }*/

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

 //   maskFilter->SetMaskImage(dilateFilter->GetOutput());

    // upsample the binary image
    typedef itk::ResampleImageFilter<TFloatImage, TFloatImage> resampleFilterType;
    typename resampleFilterType::Pointer resampleBinaryFilter = resampleFilterType::New();
    resampleBinaryFilter->SetInput( dilateFilter->GetOutput() );

    resampleBinaryFilter->SetOutputOrigin( dilateFilter->GetOutput()->GetOrigin() );

    InputSizeType outputBinarySize = dilateFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
    InputSpacingType outputBinarySpacing = dilateFilter->GetOutput()->GetSpacing();
    for( int iI = 0; iI < 3; iI++ )
    {
        outputBinarySize[iI] *= 2;
        outputBinarySpacing[iI] /= 2;
    }
    resampleBinaryFilter->SetSize( outputBinarySize );
    resampleBinaryFilter->SetOutputSpacing( outputBinarySpacing );

    typedef itk::IdentityTransform<double, 3> TransformType;
    resampleBinaryFilter->SetTransform( TransformType::New() );

    resampleBinaryFilter->Update();
    maskFilter->SetMaskImage( resampleBinaryFilter->GetOutput() );
    }

  // upsample the original image
  typedef itk::ResampleImageFilter<TInputImage, TInputImage> resampleFilterType;
  typename resampleFilterType::Pointer resampleOriginalFilter = resampleFilterType::New();
  resampleOriginalFilter->SetInput( this->GetInput() );

  resampleOriginalFilter->SetOutputOrigin( this->GetInput()->GetOrigin() );

  InputSizeType outputOriginalSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
  InputSpacingType outputOriginalSpacing = this->GetInput()->GetSpacing();
  for( int iI = 0; iI < 3; iI++ )
  {
        outputOriginalSize[iI] *= 2;
        outputOriginalSpacing[iI] /= 2;
  }
  resampleOriginalFilter->SetSize( outputOriginalSize );
  resampleOriginalFilter->SetOutputSpacing( outputOriginalSpacing );

  typedef itk::IdentityTransform<double, 3> TransformType;
  resampleOriginalFilter->SetTransform( TransformType::New() );

  typedef itk::BSplineInterpolateImageFunction<TInputImage, double, double> bsplineInterpolatorType;
  typename bsplineInterpolatorType::Pointer interpolatorOriginal = bsplineInterpolatorType::New();
  interpolatorOriginal->SetSplineOrder(3);
  resampleOriginalFilter->SetInterpolator( interpolatorOriginal );

  resampleOriginalFilter->Update();

  /*{
  typedef itk::ImageFileWriter<TInputImage> WriterLabelType;
    typename WriterLabelType::Pointer writerSample = WriterLabelType::New();
    writerSample->SetInput( resampleOriginalFilter->GetOutput() );
    std::string filename = "/playpen/Project/airwayAtlas/AirwaySegmenterInSlicer";
    filename += "/upsample_original.nhdr";
    writerSample->SetFileName( filename );

    try
      {
      writerSample->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl;
      }
   }*/

  //
  //Masking
  //
  
  //maskFilter->SetInput(this->GetInput());
  maskFilter->SetInput(resampleOriginalFilter->GetOutput());
  maskFilter->SetOutsideValue(std::numeric_limits<InputPixelType>::max());
  maskFilter->Update();

  /*
  // Upsampling 
  typedef itk::ResampleImageFilter<TInputImage, TInputImage> resampleFilterType;
  typename resampleFilterType::Pointer resampleFilter = resampleFilterType::New();
  resampleFilter->SetInput( maskFilter->GetOutput() );

  resampleFilter->SetOutputOrigin( maskFilter->GetOutput()->GetOrigin() );

  InputSizeType outputSize = maskFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
  InputSpacingType outputSpacing = maskFilter->GetOutput()->GetSpacing();
  for( int iI = 0; iI < 3; iI++ )
  {
	outputSize[iI] *= 2;
	outputSpacing[iI] /= 2;
  }
  resampleFilter->SetSize( outputSize );
  resampleFilter->SetOutputSpacing( outputSpacing );

  typedef itk::IdentityTransform<double, 3> TransformType;
  resampleFilter->SetTransform( TransformType::New() );

  typedef itk::BSplineInterpolateImageFunction<TInputImage, double, double> bsplineInterpolatorType;
  typename bsplineInterpolatorType::Pointer interpolator = bsplineInterpolatorType::New();
  interpolator->SetSplineOrder(3);
  resampleFilter->SetInterpolator( interpolator );

  resampleFilter->Update();
  
  //if (bDebug)
  //  {
    typedef itk::ImageFileWriter<TInputImage> WriterLabelType;
    typename WriterLabelType::Pointer writerSample = WriterLabelType::New();
    writerSample->SetInput( resampleFilter->GetOutput() );
    std::string filename = "/playpen/Project/airwayAtlas/AirwaySegmenterInSlicer";
    filename += "/upsample_image.nhdr";
    writerSample->SetFileName( filename );

    try
      {
      writerSample->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl; 
      }
    //}
   */

  //
  //Contour
  //
  
  //Conversion to VTK
  typedef itk::ImageToVTKImageFilter<TInputImage> adaptatorFromITKtoVTKType;
  typename adaptatorFromITKtoVTKType::Pointer toVTKFilter = adaptatorFromITKtoVTKType::New();
  toVTKFilter->SetInput( maskFilter->GetOutput() );
  //toVTKFilter->SetInput( resampleFilter->GetOutput() );
  toVTKFilter->Update();

  //Apply VTK Filter
  vtkSmartPointer<vtkContourFilter> contourFilter = vtkContourFilter::New();
  contourFilter->SetInput(toVTKFilter->GetOutput());
  contourFilter->SetValue(0, this->m_dThreshold);
  contourFilter->Update();

  // Get the largest connectivity
  vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter = vtkPolyDataConnectivityFilter::New();
  connectivityFilter->SetInput(contourFilter->GetOutput());
  connectivityFilter->SetExtractionModeToLargestRegion();
  connectivityFilter->Update();
  
  /*
  // smooth meshes
  vtkSmartPointer<vtkSmoothPolyDataFilter> smoothFilter = vtkSmoothPolyDataFilter::New();
  smoothFilter->SetInput(connectivityFilter->GetOutput());
  smoothFilter->SetNumberOfIterations(100);
  smoothFilter->Update();
  */  

  //Transform it to IJK (i.e. pixel space) 
  // <-> equivalent to a rotation of 180 degrees around the Z axis
  vtkSmartPointer<vtkTransform> transform = vtkTransform::New();
  transform->RotateZ(180.0);
  
  vtkSmartPointer<vtkTransformPolyDataFilter> transformer = 
    vtkTransformPolyDataFilter::New();
  transformer->SetInput(connectivityFilter->GetOutput());
  //transformer->SetInput(smoothFilter->GetOutput());
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
