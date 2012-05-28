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

// STL includes
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <time.h>
#include <vector>


//ITK includes
#include <itkAbsoluteValueDifferenceImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkFastMarchingImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageSliceConstIteratorWithIndex.h>
#include <itkImageSliceIteratorWithIndex.h>
#include <itkIdentityTransform.h>
#include <itkLabelGeometryImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkMatrix.h>
#include <itkMultiplyImageFilter.h>
#include <itkOtsuThresholdImageFilter.h>
#include <itkRelabelComponentImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkSmartPointer.h>
#include <itkSpatialOrientationAdapter.h>
#include <itkSubtractImageFilter.h>


//Local includes
#include "AirwaySegmenterCLP.h"
#include "AirwaySegmenterConfig.h"

#include "itkAirwaySurfaceWriter.h"
#include "itkMaskedOtsuThresholdImageFilter.h"


namespace
{

int outputAllSettings(int argc, char* argv[])
{
  PARSE_ARGS;

  std::cout << "Parameter settings:" << std::endl;
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "input image                  = " << inputImage << std::endl;
  std::cout << "output image                 = " << outputImage << std::endl;
  std::cout << "output geometry              = " << outputGeometry << std::endl;
  std::cout << std::endl;
  std::cout << "dMaxAirwayRadius             = " << dMaxAirwayRadius << std::endl;
  std::cout << "dErodeDistance               = " << dErodeDistance << std::endl;
  std::cout << "iMaximumNumberOfCVIterations = " << iMaximumNumberOfCVIterations << std::endl;
  std::cout << "dCVLambda                    = " << dCVLambda << std::endl;
  std::cout << "iComponent                   = " << iComponent << std::endl;
  std::cout << "lowerSeed                    = " << lowerSeed[0]
                                             << ", " << lowerSeed[1]
                                             << ", " << lowerSeed[2]
                                             << std::endl;
  std::cout << "lowerSeedRadius              = " << lowerSeedRadius << std::endl;
  std::cout << "upperSeed                    = " << upperSeed[0]
                                             << ", " << upperSeed[1]
                                             << ", " << upperSeed[2]
                                             << std::endl;
  std::cout << "upperSeedRadius         = " << upperSeedRadius << std::endl;
  std::cout << "bNoWarning                   = " << bNoWarning <<std::endl;
  std::cout << "bDebug                       = " << bDebug << std::endl;
  std::cout << "sDebugFolder                 = " << sDebugFolder << std::endl;
  std::cout << "bRAIImage                    = " << bRAIImage << std::endl;
  std::cout << "sRAIImagePath                = " << sRAIImagePath << std::endl;
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << std::endl << std::endl << std::endl;

  return 0;
}

//This Fast marching is factorised for preventing the memory to 
//be overwhelmed by unused nodes. Now that 
//This function is for internal use, it assumes all 
// its parameters are given correctly

template<class T> itk::Image<float, 3>::Pointer
  FastMarchIt(typename itk::Image<typename T, 3>::Pointer image, 
              std::string type, double erodedDistance, double airwayRadius)
{
  //Necessaries typedefs
  typedef itk::Image<float, 3> FloatImageType;
  typedef itk::Image<typename T, 3> LabelImageType;
  typedef itk::FastMarchingImageFilter< FloatImageType, FloatImageType >  FastMarchingFilterType;
  typedef FastMarchingFilterType::NodeContainer  NodeContainer;
  typedef FastMarchingFilterType::NodeType NodeType;
  typedef itk::ImageRegionConstIterator< LabelImageType > ConstIteratorType; 

  //Instantiations
  FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();
  NodeContainer::Pointer seeds = NodeContainer::New();
  seeds->Initialize();

  //Nodes are created as stack variables and 
  //initialized with a value and an itk::Index position. NodeType node;
  NodeType node;
  // seed value is 0 for all of them, because these are all starting nodes
  node.SetValue( 0.0 ); 

  // loop through the output image 
  // and set all voxels to 0 seed voxels
  ConstIteratorType it( image, image->GetLargestPossibleRegion() );

  unsigned int uiNumberOfSeeds = 0;
  for ( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
    if (type.compare("Out") == 0) //Dilatation
      {
      if ( it.Get() > 0 )
        {
        node.SetIndex( it.GetIndex() );
        seeds->InsertElement( uiNumberOfSeeds++, node );
        }
      }
    else if (type.compare("In") == 0)//Erosion
      {
      if ( it.Get() == 0 )
        {
        node.SetIndex( it.GetIndex() );
        seeds->InsertElement( uiNumberOfSeeds++, node );
        }
      }
    }

  //The set of seed nodes is now passed to the
  // FastMarchingImageFilter with the method SetTrialPoints().
  fastMarching->SetTrialPoints( seeds );

  // The FastMarchingImageFilter requires the user to specify 
  //the size of the image to be produced as output.
  //This is done using the SetOutputSize().

  fastMarching->SetInput( NULL );
  fastMarching->SetSpeedConstant( 1.0 );  // to solve a simple Eikonal equation

  fastMarching->SetOutputSize( image->GetBufferedRegion().GetSize() );
  fastMarching->SetOutputRegion( image->GetBufferedRegion() );
  fastMarching->SetOutputSpacing( image->GetSpacing() );
  fastMarching->SetOutputOrigin( image->GetOrigin() );

  fastMarching->SetStoppingValue( airwayRadius + erodedDistance + 1 );

  try
    {
    fastMarching->Update();
    }
  catch(itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl; 
    }

  return fastMarching->GetOutput();
}

//Look for the labels in the sphercal region within radius of
//the ball center and return the most represented label

template<class T> int 
  LabelIt(typename itk::Image<typename T, 3>::Pointer image,
          std::vector<float> ballCenter, double radius, bool printLabels)
{
  typedef itk::Image<typename T, 3> TImage;
  typedef itk::Image<typename T, 3>::SizeType TSize;
  typedef itk::Image<typename T, 3>::SpacingType TSpacing;
  typedef itk::Image<typename T, 3>::PointType TOrigin;
  typedef itk::Image<typename T, 3>::IndexType TIndex;

  TOrigin imageOrigin = image->GetOrigin();
  TSpacing imageSpacing = image->GetSpacing();
  TSize imageSize = image->GetBufferedRegion().GetSize();

  // Convert to LPS system
  float x, y, z;
  x = -ballCenter[0];
  y = -ballCenter[1];
  z =  ballCenter[2];

  //Get the bounding box around the piryna aperture
  int region[6];
  region[0] = int( floor( ( x - radius - imageOrigin[0] ) / imageSpacing[0] ) );
  region[1] = int( floor( ( y - radius - imageOrigin[1] ) / imageSpacing[1] ) );
  region[2] = int( floor( ( z - radius - imageOrigin[2] ) / imageSpacing[2] ) );
  region[3] = int(  ceil( ( x + radius - imageOrigin[0] ) / imageSpacing[0] ) );
  region[4] = int(  ceil( ( y + radius - imageOrigin[1] ) / imageSpacing[1] ) );
  region[5] = int(  ceil( ( z + radius - imageOrigin[2] ) / imageSpacing[2] ) );

  region[0] = region[0] > 0 ? region[0] : 0;
  region[1] = region[1] > 0 ? region[1] : 0;
  region[2] = region[2] > 0 ? region[2] : 0;

  region[3] = region[3] < (imageSize[0]-1) ? region[3] : (imageSize[0]-1);
  region[4] = region[4] < (imageSize[1]-1) ? region[4] : (imageSize[1]-1);
  region[5] = region[5] < (imageSize[2]-1) ? region[5] : (imageSize[2]-1);

  if (printLabels)
    {
    std::cout << "Region: " << region[0] << ", " 
              << region[1] << ", " << region[2] << ", "
              << region[3] << ", " << region[4] << ", " 
              << region[5] << std::endl; 
    }

  std::map<int, int> labels;

  for( int iI=region[0]; iI<=region[3]; iI++ )
    {
    for( int iJ=region[1]; iJ<=region[4]; iJ++ )
      {
      for( int iK=region[2]; iK<=region[5]; iK++ )
        {
        //Get real space position
        double iX = iI * imageSpacing[0] + imageOrigin[0] - x;
        double iY = iJ * imageSpacing[1] + imageOrigin[1] - y;
        double iZ = iK * imageSpacing[2] + imageOrigin[2] - z;

        //If within the nose ball region
        if( iX * iX + iY * iY + iZ * iZ <= radius * radius )
          {
          TIndex pixelIndex;
          pixelIndex[0] = iI;
          pixelIndex[1] = iJ;
          pixelIndex[2] = iK;

          int label = image->GetPixel( pixelIndex );
          
          if( label != 0 )//ignore label 0, it's always the background
            {
            labels[label] += 1;
            }
          }
        }
      }
    }

  int finalLabel = 0;
  int labelCount = 0;
  for(std::map<int, int>::const_iterator it = labels.begin();
      it != labels.end(); ++it)
    {
    if (printLabels)
      {
      std::cout<<"Labels "<<it->first<<" :   "<<it->second<<std::endl;
      }

    if (it->second > labelCount)
      {
      labelCount = it->second;
      finalLabel = it->first;
      }
    }

  return finalLabel;
}

template<class T> int DoIt(int argc, char* argv[], T)
{
  //--
  //-- Typedefs
  //--
  
  typedef float TFloatType;
  typedef T TPixelType;
  typedef T TLabelPixelType;

  const unsigned char DIMENSION = 3;

  typedef itk::Image< TPixelType,DIMENSION> InputImageType;
  typedef itk::Image< TPixelType,DIMENSION> OutputImageType;
  typedef itk::Image< TLabelPixelType,DIMENSION> LabelImageType;
  typedef itk::Image< TFloatType,DIMENSION> FloatImageType;
  typedef itk::Image< unsigned char,DIMENSION> UCharImageType;

  typedef itk::ImageFileReader< InputImageType > ReaderType;
  typedef itk::ImageFileReader< LabelImageType > ReaderLabelType;
  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  typedef itk::ImageFileWriter< LabelImageType > WriterLabelType;

  typedef LabelImageType::SizeType TSize;
  typedef LabelImageType::SpacingType TSpacing;
  typedef LabelImageType::PointType TOrigin;
  typedef LabelImageType::IndexType TIndex;

  //--
  //-- Parsing the input
  //--

  // parse the input arguments
  PARSE_ARGS;  

  // output the arguments
  if (bDebug)
    {
    outputAllSettings( argc, argv );
    }

  //Outputing debug result to current folder if not precised otherwise
  if(bDebug && 
      (sDebugFolder.compare("None") == 0 || sDebugFolder.compare("") == 0))
    {
    sDebugFolder = ".";
    }

  // read the input image
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputImage );

  try
    {
    reader->Update();  
    }
  catch ( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl; 
    }


  //--
  //-- Automatic Resampling to RAI
  //--
  InputImageType::Pointer originalImage = reader->GetOutput();
  InputImageType::DirectionType originalImageDirection = originalImage->GetDirection();

  itk::SpatialOrientationAdapter adapter;
  InputImageType::DirectionType RAIDirection =
    adapter.ToDirectionCosines(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI);

  bool shouldConvert = false;
  for (int i = 0; i < 3; ++i)
    {
    for (int j = 0; j < 3; ++j)
      {
      if (abs(originalImageDirection[i][j] - RAIDirection[i][j]) > 1e-6)
        {
        shouldConvert = true;
        break;
        }
      }
    }

  typedef itk::ResampleImageFilter<InputImageType, InputImageType> ResampleImageFilterType;
  ResampleImageFilterType::Pointer resampleFilter = ResampleImageFilterType::New();

  if (shouldConvert)
    {
    typedef itk::IdentityTransform<double, DIMENSION> IdentityTransformType;

    resampleFilter->SetTransform(IdentityTransformType::New());
    resampleFilter->SetInput(originalImage);
    resampleFilter->SetSize(originalImage->GetLargestPossibleRegion().GetSize());
    resampleFilter->SetOutputOrigin(originalImage->GetOrigin());
    resampleFilter->SetOutputSpacing(originalImage->GetSpacing());
    resampleFilter->Update();

    originalImage = resampleFilter->GetOutput();
    }

  //Write RAI Image if asked to
  if (bRAIImage)
    {
    WriterType::Pointer writer = WriterType::New();
    writer->SetInput( originalImage);
    writer->SetFileName( sRAIImagePath.c_str() );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      }
    }

  //--
  //-- Otsu thresholding first
  //--

  typedef itk::OtsuThresholdImageFilter< InputImageType, LabelImageType > OtsuThresholdFilterType;
  OtsuThresholdFilterType::Pointer otsuThresholdFilter = OtsuThresholdFilterType::New();

  otsuThresholdFilter->SetInsideValue( 0 );
  otsuThresholdFilter->SetOutsideValue( 1 );
  otsuThresholdFilter->SetInput( originalImage );
  otsuThresholdFilter->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( otsuThresholdFilter->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/otsu.nhdr";
    writer->SetFileName( filename );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl; 
      }
    }

  //--
  //-- Dilatation
  //--

  typedef itk::BinaryThresholdImageFilter< FloatImageType, LabelImageType > ThresholdingFilterType;
  ThresholdingFilterType::Pointer thresholdDilation = ThresholdingFilterType::New();

  thresholdDilation->SetLowerThreshold( 0.0 );
  thresholdDilation->SetUpperThreshold( dMaxAirwayRadius );
  thresholdDilation->SetOutsideValue( 0 );
  thresholdDilation->SetInsideValue( 1 );

  //Using custom fast marching function
  thresholdDilation->SetInput( FastMarchIt<typename T>(
                                otsuThresholdFilter->GetOutput(),
                                "Out",
                                dErodeDistance, 
                                dMaxAirwayRadius) );

  thresholdDilation->Update();

  // Now write this for test purposes
  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( thresholdDilation->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/fmt-out.nhdr";
    writer->SetFileName( filename );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl; 
      }
    }

  //--
  //-- Erosion (Thus creating a closing)
  //--

  ThresholdingFilterType::Pointer thresholdClosing = ThresholdingFilterType::New();

  thresholdClosing->SetLowerThreshold( 0.0 ); 
  thresholdClosing->SetUpperThreshold( dMaxAirwayRadius );
  thresholdClosing->SetOutsideValue( 1 ); 
  thresholdClosing->SetInsideValue( 0 );

  thresholdClosing->SetInput( FastMarchIt<typename T>(
                                thresholdDilation->GetOutput(),
                                "In",
                                dErodeDistance,
                                dMaxAirwayRadius) );
  thresholdClosing->Update();

  // Now write this for test purposes
  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( thresholdClosing->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/fmtIn-out.nhdr";
    writer->SetFileName( filename );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl; 
      }
    }

  //--
  //-- Difference between closed image and ostu-threshold of the original one
  //--

  typedef itk::AbsoluteValueDifferenceImageFilter< LabelImageType, LabelImageType, LabelImageType >
    TAbsoluteValueDifferenceFilter;
  TAbsoluteValueDifferenceFilter::Pointer absoluteValueDifferenceFilter 
    = TAbsoluteValueDifferenceFilter::New();

  absoluteValueDifferenceFilter->SetInput1( otsuThresholdFilter->GetOutput() );
  absoluteValueDifferenceFilter->SetInput2( thresholdClosing->GetOutput() );
  absoluteValueDifferenceFilter->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( absoluteValueDifferenceFilter->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/avd.nhdr";
    writer->SetFileName( filename );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      }
    }

  //--
  //-- Create a sligthly eroded version of the closed image
  //-- This is to prevent any weird effects at the outside of the face
  //--

  // We can get this simply by taking a
  // slightly different threshold for the marching in case

  ThresholdingFilterType::Pointer thresholdDifference = ThresholdingFilterType::New();

  thresholdDifference->SetLowerThreshold( 0.0 );
  thresholdDifference->SetUpperThreshold( dMaxAirwayRadius+dErodeDistance );
  thresholdDifference->SetOutsideValue( 1 );
  thresholdDifference->SetInsideValue( 0 );

  //The closed image was the input of the closing threshold
  // (we don't want to re-run a fast marching)
  thresholdDifference->SetInput( thresholdClosing->GetInput() );

  // now do the masking
  typedef itk::MaskImageFilter< LabelImageType, LabelImageType, LabelImageType > TMaskImageFilter;
  TMaskImageFilter::Pointer absoluteValueDifferenceFilterMasked 
    = TMaskImageFilter::New();

  absoluteValueDifferenceFilterMasked->SetInput1(
    absoluteValueDifferenceFilter->GetOutput() );

  absoluteValueDifferenceFilterMasked->SetInput2(
    thresholdDifference->GetOutput() ); // second input is the mask

  //--
  //-- Extract largest component of the difference
  //--

  //(which should be -- hopefully -- the airway,
  // since it was adapted to the expected size)

  if (bDebug)
    {
    std::cout << "Extracting largest connected component ... ";
    }

  typedef itk::ConnectedComponentImageFilter< LabelImageType, LabelImageType > ConnectedComponentType;
  typedef itk::RelabelComponentImageFilter< LabelImageType, LabelImageType > RelabelComponentType;
  typedef itk::BinaryThresholdImageFilter< LabelImageType, LabelImageType > FinalThresholdingFilterType;

  ConnectedComponentType::Pointer connected = ConnectedComponentType::New();
  RelabelComponentType::Pointer relabel = RelabelComponentType::New();
  FinalThresholdingFilterType::Pointer largestComponentThreshold = FinalThresholdingFilterType::New();

  // Label the components in the image and relabel them so that object
  // numbers increase as the size of the objects decrease.
  connected->SetInput ( absoluteValueDifferenceFilterMasked->GetOutput());
  relabel->SetInput( connected->GetOutput() );
  relabel->SetNumberOfObjectsToPrint( 5 );
  relabel->Update();

  int componentNumber = 0;
  if (iComponent <= 0)
    {
    componentNumber = LabelIt<typename T>(relabel->GetOutput(),
                                            lowerSeed,
                                            lowerSeedRadius,
                                            bDebug);
    //std::cout<<"Label found = "<<componentNumber<<std::endl;
    }
  else
    {
    componentNumber = iComponent;
    }

  // pull out the largest object
  largestComponentThreshold->SetInput( relabel->GetOutput() );
  largestComponentThreshold->SetLowerThreshold( componentNumber ); // object #1
  largestComponentThreshold->SetUpperThreshold( componentNumber ); // object #1
  largestComponentThreshold->SetInsideValue(1);
  largestComponentThreshold->SetOutsideValue(0);
  largestComponentThreshold->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer lccWriter = WriterLabelType::New();
    lccWriter->SetInput( largestComponentThreshold->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/lcc.nhdr";
    lccWriter->SetFileName( filename );

    try
      {
      lccWriter->Update();
      }
    catch( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      }

    std::cout << "done." << std::endl;
    }

  //--
  //-- Now do another Otsu thresholding but just around the current segmentation
  //-- 

  // For this, we need to make it first a little bit bigger

  ThresholdingFilterType::Pointer thresholdExtendedSegmentation = ThresholdingFilterType::New();

  thresholdExtendedSegmentation->SetInput( FastMarchIt<typename T>(
                                              largestComponentThreshold->GetOutput(),
                                              "Out", 
                                              dErodeDistance,
                                              dMaxAirwayRadius) );

  thresholdExtendedSegmentation->SetLowerThreshold( 0.0 );
  
  // to make sure we get roughly twice the volume if the object would have a circular
  // cross section;
  // TODO: maybe make this a command line argument?
  thresholdExtendedSegmentation->SetUpperThreshold( (sqrt(2.0)-1)*dMaxAirwayRadius );

  thresholdExtendedSegmentation->SetOutsideValue( 0 ); 
  thresholdExtendedSegmentation->SetInsideValue( 1 );

  // force it so we have the mask available
  thresholdExtendedSegmentation->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( thresholdExtendedSegmentation->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/seg-extended.nhdr";
    writer->SetFileName( filename );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl; 
      }
    }

  // now do another Otsu thresholding
  // but restrict the statistics to the currently obtained area 
  //(custom ostu-threshold filter)

  typedef itk::MaskedOtsuThresholdImageFilter< InputImageType, LabelImageType, LabelImageType >
    MaskedOtsuThresholdFilterType;
  MaskedOtsuThresholdFilterType::Pointer maskedOtsuThresholdFilter =
    MaskedOtsuThresholdFilterType::New();

  // TODO: not sure about these inside/outside settings, check!!
  maskedOtsuThresholdFilter->SetInsideValue( 1 );
  maskedOtsuThresholdFilter->SetOutsideValue( 0 );
  maskedOtsuThresholdFilter->SetMaskImage(
                              thresholdExtendedSegmentation->GetOutput() );
  maskedOtsuThresholdFilter->SetInput( originalImage );
  maskedOtsuThresholdFilter->Update();

  // write it out to see if it worked (if it did clean up the code)

  // write it out

  if (bDebug)
    {
    WriterLabelType::Pointer labelWriterMaskedOtsu = WriterLabelType::New();
    labelWriterMaskedOtsu->SetInput( maskedOtsuThresholdFilter->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/otst-out-masked.nhdr";
    labelWriterMaskedOtsu->SetFileName( filename );

    try
      {
      labelWriterMaskedOtsu->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      }
    }

  //--
  //-- Now mask it again and extract the largest component
  //--

  if (bDebug)
    {
    std::cout << " mask it again and extract the largest component ... " << std::endl;
    }

  TMaskImageFilter::Pointer maskedOtsu = TMaskImageFilter::New();

  maskedOtsu->SetInput1( maskedOtsuThresholdFilter->GetOutput() );
  maskedOtsu->SetInput2( thresholdDifference->GetOutput() ); // second input is the mask

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( maskedOtsu->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/maskedOtsu-out_second.nhdr";
    writer->SetFileName( filename );

    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl;
      }

    std::cout << " Update ... " << std::endl;
    }

  try
    {
    maskedOtsu->Update();
    }
  catch ( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }

  //--
  //-- extract largest the airway with the lungs
  //--

  if (bDebug)
    {
    std::cout << "Extracting final largest connected component ... ";
    }

  ConnectedComponentType::Pointer connectedFinal = ConnectedComponentType::New();
  RelabelComponentType::Pointer relabelFinal = RelabelComponentType::New();
  FinalThresholdingFilterType::Pointer finalThreshold = FinalThresholdingFilterType::New();

  // Label the components in the image and relabel them so that object
  // numbers increase as the size of the objects decrease.
  connectedFinal->SetInput ( maskedOtsu->GetOutput());
  relabelFinal->SetInput( connectedFinal->GetOutput() );
  relabelFinal->SetNumberOfObjectsToPrint( 5 );
  relabelFinal->Update();

  if (iComponent <= 0)
    {
    componentNumber = LabelIt<typename T>(relabelFinal->GetOutput(),
                            lowerSeed,
                            lowerSeedRadius,
                            bDebug);
    //std::cout<<"Label found = "<<componentNumber<<std::endl;
    }
  else
    {
    componentNumber = iComponent;
    }

  // pull out the largest object
  finalThreshold->SetInput( relabelFinal->GetOutput() );
  finalThreshold->SetLowerThreshold( componentNumber ); // object #1
  finalThreshold->SetUpperThreshold( componentNumber ); // object #1
  finalThreshold->SetInsideValue(1);
  finalThreshold->SetOutsideValue(0);
  finalThreshold->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( finalThreshold->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/airway-with-lung.nrrd";
    writer->SetFileName( filename );
    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      }

    std::cout<<"..done"<<std::endl;
    }

  //--
  //--
  //Second part of the code : getting rid of lung automatically
  //--
  //--

  TOrigin imageOrigin = originalImage->GetOrigin();
  TSpacing imageSpacing = originalImage->GetSpacing();
  TSize imageSize = originalImage->GetBufferedRegion().GetSize();

  //--
  // Need to convert from RAS (Slicer) to ITK's coordinate system: LPS 
  //--

  float ballX, ballY, ballZ;
  ballX = -lowerSeed[0];
  ballY = -lowerSeed[1];
  ballZ = lowerSeed[2];
  
  if (bDebug)
    {
    std::cout << "(x, y, z): " << ballX 
              << " " << ballY 
              << " " << ballZ 
              << ", Radius: " 
              << lowerSeedRadius << std::endl;
    }

  //--
  //-- Compute the cubic region around the ball
  //--

  int ballRegion[6];
  ballRegion[0] = int(floor( ( ballX - lowerSeedRadius - imageOrigin[0] ) / imageSpacing[0] ));
  ballRegion[1] = int(floor( ( ballY - lowerSeedRadius - imageOrigin[1] ) / imageSpacing[1] ));
  ballRegion[2] = int(floor( ( ballZ - lowerSeedRadius - imageOrigin[2] ) / imageSpacing[2] ));
  ballRegion[3] = int(ceil( ( ballX + lowerSeedRadius - imageOrigin[0] ) / imageSpacing[0] ));
  ballRegion[4] = int(ceil( ( ballY + lowerSeedRadius - imageOrigin[1] ) / imageSpacing[1] ));
  ballRegion[5] = int(ceil( ( ballZ + lowerSeedRadius - imageOrigin[2] ) / imageSpacing[2] ));

  ballRegion[0] = ballRegion[0] > 0 ? ballRegion[0] : 0;
  ballRegion[1] = ballRegion[1] > 0 ? ballRegion[1] : 0;
  ballRegion[2] = ballRegion[2] > 0 ? ballRegion[2] : 0;

  ballRegion[3] = ballRegion[3] < (imageSize[0]-1) ? ballRegion[3] : (imageSize[0]-1);
  ballRegion[4] = ballRegion[4] < (imageSize[1]-1) ? ballRegion[4] : (imageSize[1]-1);
  ballRegion[5] = ballRegion[5] < (imageSize[2]-1) ? ballRegion[5] : (imageSize[2]-1); 

  if (bDebug)
    {
    std::cout << "Origin: "  << imageOrigin[0] << " "
              << imageOrigin[1] << " " 
              << imageOrigin[2] << std::endl;
    std::cout << "Spacing: " << imageSpacing[0]
              << " " << imageSpacing[1]
              << " " << imageSpacing[2] << std::endl;
    std::cout << "size: "    << imageSize[0]
              << " " << imageSize[1] << " "
              << imageSize[2] << std::endl;
    std::cout << "Ball Region: "  << ballRegion[0] << " "
                             << ballRegion[1] << " "
                             << ballRegion[2] << " "
                             << ballRegion[3] << " "
                             << ballRegion[4] << " "
                             << ballRegion[5] << std::endl;
    }

  //--
  //--
  //--

  InputImageType::Pointer imageBranch = InputImageType::New();
  InputImageType::SizeType sizeBranch;
  sizeBranch[0] = ballRegion[3]-ballRegion[0]+1;
  sizeBranch[1] = ballRegion[4]-ballRegion[1]+1;
  sizeBranch[2] = ballRegion[5]-ballRegion[2]+1;
  InputImageType::IndexType startBranch;
  startBranch.Fill(0);
  InputImageType::RegionType regionBranch;
  regionBranch.SetSize( sizeBranch );
  regionBranch.SetIndex( startBranch );
  imageBranch->SetRegions( regionBranch );

  try
    {
    imageBranch->Allocate();
    }
  catch (itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    std::cerr << "Please verify your parameters, this is often caused "
              << "by a misplaced trachea carina"<<std::endl;
    }

  for( int iI=ballRegion[0]; iI<=ballRegion[3]; iI++ )
    {
    for( int iJ=ballRegion[1]; iJ<=ballRegion[4]; iJ++ )
      {
      for( int iK=ballRegion[2]; iK<=ballRegion[5]; iK++ )
        {
        double iX = iI * imageSpacing[0] + imageOrigin[0] - ballX;
        double iY = iJ * imageSpacing[1] + imageOrigin[1] - ballY;
        double iZ = iK * imageSpacing[2] + imageOrigin[2] - ballZ;
        TIndex pixelIndexBranch;
        pixelIndexBranch[0] = iI - ballRegion[0];
        pixelIndexBranch[1] = iJ - ballRegion[1];
        pixelIndexBranch[2] = iK - ballRegion[2];
        imageBranch->SetPixel( pixelIndexBranch, 0 );

        //If within the radius of the ball
        if( iX * iX + iY * iY + iZ * iZ <= lowerSeedRadius * lowerSeedRadius )
          {
          TIndex pixelIndex;
          pixelIndex[0] = iI;
          pixelIndex[1] = iJ;
          pixelIndex[2] = iK;

          if( finalThreshold->GetOutput()->GetPixel(pixelIndex) )
            {
            finalThreshold->GetOutput()->SetPixel(pixelIndex, 0);
            imageBranch->SetPixel( pixelIndexBranch, 1 );
            }
          }
        }
      }
    }

  //--
  //-- Pick out the branch part
  //--

  ConnectedComponentType::Pointer connectedBranch = ConnectedComponentType::New();
  RelabelComponentType::Pointer relabelBranch = RelabelComponentType::New();

  connectedBranch->SetInput( imageBranch );
  connectedBranch->Update();
  relabelBranch->SetInput( connectedBranch->GetOutput() );
  relabelBranch->SetNumberOfObjectsToPrint( 5 );
  relabelBranch->Update();

  //--
  //-- Get geometry statistics
  //--

  typedef itk::LabelGeometryImageFilter<LabelImageType> LabelGeometryImageFilterType;
  LabelGeometryImageFilterType::Pointer labelBranchGeometry =
    LabelGeometryImageFilterType::New();
  labelBranchGeometry->SetInput( relabelBranch->GetOutput() );
  labelBranchGeometry->CalculateOrientedBoundingBoxOn();
  //just in case
  labelBranchGeometry->CalculateOrientedLabelRegionsOff();
  labelBranchGeometry->CalculatePixelIndicesOff();
  labelBranchGeometry->CalculateOrientedIntensityRegionsOff();
  labelBranchGeometry->Update();

  int nBranchParts = relabelBranch->GetNumberOfObjects();
  int nBranchId = 1;
  if( nBranchParts > 1 )
    {
    if (bDebug)
      {
      std::cout << "number of parts in branch: " << nBranchParts << std::endl;
      }

    double minDist2Ball;
    int minLabel;

    double dBallIndexX = ( ballRegion[3] - ballRegion[0] ) / 2.0;
    double dBallIndexY = ( ballRegion[4] - ballRegion[1] ) / 2.0;
    double dBallIndexZ = ( ballRegion[5] - ballRegion[2] ) / 2.0;

    for( int nNumParts=1; nNumParts<=nBranchParts; nNumParts++ )
      {
      LabelGeometryImageFilterType::BoundingBoxType boundingBox =
         labelBranchGeometry->GetBoundingBox( nNumParts );
      double xTmp = ( boundingBox[0] + boundingBox[1] ) / 2.0 - dBallIndexX;
      double yTmp = ( boundingBox[2] + boundingBox[3] ) / 2.0 - dBallIndexY;
      double zTmp = ( boundingBox[4] + boundingBox[5] ) / 2.0 - dBallIndexZ;
      double distTmp = sqrt( xTmp * xTmp + yTmp * yTmp + zTmp * zTmp );

      if (bDebug)
        {
        std::cout << "( " << xTmp 
                  << ", " << yTmp 
                  << ", " << zTmp 
                  << "), " << distTmp 
                  << std::endl;
        }

      if( nNumParts == 1 || minDist2Ball > distTmp)
        {
        minDist2Ball = distTmp;
        minLabel = nNumParts;
        }

      if (bDebug)
        {
        std::cout << boundingBox << std::endl;
        }

      }
    nBranchId = minLabel;
    }
  

  FinalThresholdingFilterType::Pointer branchThreshold = 
    FinalThresholdingFilterType::New();
  branchThreshold->SetInput( relabelBranch->GetOutput() );
  branchThreshold->SetLowerThreshold( nBranchId );
  branchThreshold->SetUpperThreshold( nBranchId );
  branchThreshold->SetInsideValue(1);
  branchThreshold->SetOutsideValue(0);
  branchThreshold->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( branchThreshold->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/FinalThresholdNoBranch.nrrd";
    writer->SetFileName( filename );
    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl;
      }
    }

  if (bDebug)
    {
    std::cout << "Final airway label ... " << std::endl;
    }

  ConnectedComponentType::Pointer connectedFinalWithoutLung = ConnectedComponentType::New();
  RelabelComponentType::Pointer relabelFinalWithoutLung = RelabelComponentType::New();

  connectedFinalWithoutLung->SetInput( finalThreshold->GetOutput() );
  relabelFinalWithoutLung->SetInput( connectedFinalWithoutLung->GetOutput() );
  relabelFinalWithoutLung->SetNumberOfObjectsToPrint( 5 );
  relabelFinalWithoutLung->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( relabelFinalWithoutLung->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/relabelFinalWithLung.nrrd";
    writer->SetFileName( filename );
    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl;
      }
    std::cout << "Relabeled" << std::endl;
    }

  //--
  //--Find the airway label using the pyryna apreture position
  //--

  int nNumAirway = 0;
  if (iComponent <= 0)
    {
    nNumAirway = LabelIt<typename T>(relabelFinalWithoutLung->GetOutput(),
                            upperSeed,
                            upperSeedRadius,
                            bDebug);
    std::cout<<"Label found = "<<componentNumber<<std::endl;
    }
  else
    {
    componentNumber = nNumAirway;
    }

  //Check if the maximum label found is 0,
  // meaning that no label was found n the nose region
  //
  //-> Nasal cavity probably not segmented !
  //
  if (nNumAirway == 0) 
    {
    std::cerr<<"WARNING !"<<std::endl;
    std::cerr<<"The maximum label found in the spherical region around"
             <<" the pyrina aperture was zero !"<<std::endl
             <<"This probably means that nasal cavity is not segmented "
             <<" (or the point is misplaced)."<<std::endl
             <<" Advice: use --debug to ouput and check all the labels "
             <<" found and/or increase the upperSeedRadius to "
             <<"cover more space"
             <<std::endl;
    if (bNoWarning)
      {
      return EXIT_FAILURE;
      }
    }

  if (bDebug)
    {
    std::cout << "The label " << nNumAirway 
              << " is picked as the airway." << std::endl;
    }

  FinalThresholdingFilterType::Pointer finalAirwayThreshold = 
    FinalThresholdingFilterType::New();
  finalAirwayThreshold->SetInput( relabelFinalWithoutLung->GetOutput() );
  finalAirwayThreshold->SetLowerThreshold( nNumAirway ); 
  finalAirwayThreshold->SetUpperThreshold( nNumAirway ); 
  finalAirwayThreshold->SetInsideValue(1);
  finalAirwayThreshold->SetOutsideValue(0); 
  finalAirwayThreshold->Update();

  if (bDebug)
    {
    WriterLabelType::Pointer writer = WriterLabelType::New();
    writer->SetInput( finalAirwayThreshold->GetOutput() );
    std::string filename = sDebugFolder;
    filename += "/final_threshold.nrrd";
    writer->SetFileName( filename );
    try
      {
      writer->Update();
      }
    catch ( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl; std::cerr << excep << std::endl;
      }
    }

  // put the ball back
  if (bDebug)
    {
    std::cout << "Putting the branches back ... " << std::endl;
    }

  for( int iI=ballRegion[0]; iI<=ballRegion[3]; iI++ )
    {
    for( int iJ=ballRegion[1]; iJ<=ballRegion[4]; iJ++ )
      {
      for( int iK=ballRegion[2]; iK<=ballRegion[5]; iK++ )
        {
        double iX = iI * imageSpacing[0] + imageOrigin[0] - ballX;
        double iY = iJ * imageSpacing[1] + imageOrigin[1] - ballY;
        double iZ = iK * imageSpacing[2] + imageOrigin[2] - ballZ;

        if( iX * iX + iY * iY + iZ * iZ <= lowerSeedRadius * lowerSeedRadius )
          {
          TIndex pixelIndex;
          pixelIndex[0] = iI;
          pixelIndex[1] = iJ;
          pixelIndex[2] = iK;
          
          TIndex pixelIndexBranch;
          pixelIndexBranch[0] = iI - ballRegion[0];
          pixelIndexBranch[1] = iJ - ballRegion[1];
          pixelIndexBranch[2] = iK - ballRegion[2];

          if( branchThreshold->GetOutput()->GetPixel( pixelIndexBranch ) )
            {
            finalAirwayThreshold->GetOutput()->SetPixel(pixelIndex, 1);
            }
          }
        }
      }
    }

  if (bDebug)
    {
    std::cout << "Writing the final image ... " << std::endl;
    }

  //--
  //-- Write final image
  //--
  WriterLabelType::Pointer lccWriterFinal = WriterLabelType::New();
  lccWriterFinal->SetInput( finalAirwayThreshold->GetOutput() );
  lccWriterFinal->SetFileName( outputImage.c_str() );

  try
    {
    lccWriterFinal->Update();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    }

  //--
  //-- Write Surface
  //--

  itk::AirwaySurfaceWriter< InputImageType, LabelImageType >::Pointer surfaceWriter=
    itk::AirwaySurfaceWriter< InputImageType, LabelImageType >::New();
  surfaceWriter->SetFileName( outputGeometry.c_str() );
  surfaceWriter->SetUseFastMarching(true);
  surfaceWriter->SetMaskImage( finalAirwayThreshold->GetOutput() );
  surfaceWriter->SetInput( originalImage );

  try
    {
    surfaceWriter->Update();
    }
  catch( itk::ExceptionObject & excep )
    {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    }

  if (bDebug)
    {
    std::cout << "done." << std::endl;
    }

  return EXIT_SUCCESS;
}


void GetImageType (std::string fileName,
                  itk::ImageIOBase::IOPixelType &pixelType,
                  itk::ImageIOBase::IOComponentType &componentType)
{
  typedef itk::Image<unsigned char, 3> ImageType;
  itk::ImageFileReader<ImageType>::Pointer imageReader =
    itk::ImageFileReader<ImageType>::New();
  imageReader->SetFileName(fileName.c_str());
  imageReader->UpdateOutputInformation();

  pixelType = imageReader->GetImageIO()->GetPixelType();
  componentType = imageReader->GetImageIO()->GetComponentType();
}

}//end namespace

int main( int argc, char * argv[] )
{

  PARSE_ARGS;

#ifdef SLICER_ITK_DIFFERENT
  //HACK:
  //Used to ensure compatibility when using a different compiler than slicer
  //This will prevent from using any fancy format slicer supports
  //and not itk natively
  //
  //SHOULD NOT BE USED WHEN COMPILING WITH THE SAME COMPILER AS SLICER
  //
  char* itk_autoload_path = getenv("ITK_AUTOLOAD_PATH");
  if ( itk_autoload_path != NULL )
    {
    //Enforce no custom librairies
    putenv("ITK_AUTOLOAD_PATH=");
    }
#endif

  itk::ImageIOBase::IOPixelType     inputPixelType;
  itk::ImageIOBase::IOComponentType inputComponentType;

  int ret = EXIT_FAILURE;
  try
    {
    GetImageType(inputImage, inputPixelType, inputComponentType);

    switch( inputComponentType )
      {
      case itk::ImageIOBase::UCHAR:
        ret = DoIt( argc, argv, static_cast<unsigned char>(0) );
        break;
      case itk::ImageIOBase::CHAR:
        ret = DoIt( argc, argv, static_cast<char>(0) );
        break;
      case itk::ImageIOBase::USHORT:
        ret = DoIt( argc, argv, static_cast<unsigned short>(0) );
        break;
      case itk::ImageIOBase::SHORT:
        ret = DoIt( argc, argv, static_cast<short>(0) );
        break;
      case itk::ImageIOBase::UINT:
        ret = DoIt( argc, argv, static_cast<unsigned int>(0) );
        break;
      case itk::ImageIOBase::INT:
        ret = DoIt( argc, argv, static_cast<int>(0) );
        break;
      case itk::ImageIOBase::ULONG:
        ret = DoIt( argc, argv, static_cast<unsigned long>(0) );
        break;
      case itk::ImageIOBase::LONG:
        ret = DoIt( argc, argv, static_cast<long>(0) );
        break;
      case itk::ImageIOBase::FLOAT:
        std::cout<<"Float images not supported"<<std::endl;
        break;
      case itk::ImageIOBase::DOUBLE:
        std::cout<<"Double images not supported"<<std::endl;
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cout << "unknown component type" << std::endl;
        break;
      }
    }

  catch( itk::ExceptionObject & excep )
    {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }

  return ret;
}

