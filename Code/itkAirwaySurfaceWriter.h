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

#ifndef __itkAirwaySurfaceWriter_h
#define __itkAirwaySurfaceWriter_h

#include "itkImageToImageFilter.h"
#include <string>

namespace itk
{

/** \class AirwaySurfaceWriter 
 * \brief Given a volume and its airway segmented volume, 
write the airway surface to file
 *
 * This filter write the surface of the airway given the 
 * original image used for segmentation and the segmentation.
 * This filter can either uses fast marching (slower but more robust)
 * or the dilatation implement by itk in itkBinaryDilateImageFilter.
 * The surface is obtained by using vtkContourFilter and then written
 * to file.
 * 
 * Assume to work with 3D images.
 *
 * \sa BinaryDilateImageFilter
 * \sa FastMarchingImageFilter
 */


template<class TInputImage, class TMaskImage>
class ITK_EXPORT AirwaySurfaceWriter : 
    public ImageToImageFilter<TInputImage, TInputImage>
{
public:
  /** Standard Self typedef */
  typedef AirwaySurfaceWriter                           Self;
  typedef ImageToImageFilter<TInputImage,TInputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);  

  /** Runtime information support. */
  itkTypeMacro(IsoSurfaceWriter, ImageToImageFilter);
  
  /** Image pixel value typedef. */
  typedef typename TInputImage::PixelType   InputPixelType;

  /** Image related typedefs. */
  typedef typename TInputImage::Pointer  InputImagePointer;

  typedef typename TInputImage::SizeType    InputSizeType;
  typedef typename TInputImage::IndexType   InputIndexType;
  typedef typename TInputImage::RegionType  InputImageRegionType;

  void SetMaskImage( typename TMaskImage::ConstPointer pmask )
    {
    m_pMaskImage = pmask;
    };

  /** Image related typedefs. */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension );

  /** Get/Set the mesh output filename. */
  itkGetConstMacro(FileName, const std::string);
  itkSetMacro(FileName, std::string);

  /** Get/Set if using fast marching or not */
  itkGetConstMacro(UseFastMarching, bool);
  itkSetMacro(UseFastMarching, bool);

protected:
  AirwaySurfaceWriter();
  ~AirwaySurfaceWriter(){};
  void PrintSelf(std::ostream& os, Indent indent) const;

  void GenerateData ();

private:
  AirwaySurfaceWriter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  typename TMaskImage::ConstPointer   m_pMaskImage;
  std::string                         m_FileName;
  bool                                m_UseFastMarching;

}; // end of class

}//end namespace itk
  
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAirwaySurfaceWriter.txx"
#endif

#endif
