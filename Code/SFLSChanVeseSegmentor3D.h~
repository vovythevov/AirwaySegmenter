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

#ifndef SFLSChanVeseSegmentor3D_h_
#define SFLSChanVeseSegmentor3D_h_

#include "SFLSSegmentor3D.h"

#include <list>


template< typename TPixel >
class CSFLSChanVeseSegmentor3D : public CSFLSSegmentor3D< TPixel >
{
public:
  typedef CSFLSSegmentor3D< TPixel > SuperClassType;

  //    typedef boost::shared_ptr< CSFLSChanVeseSegmentor3D< TPixel > > Pointer;

  typedef typename SuperClassType::NodeType NodeType;
  typedef typename SuperClassType::CSFLSLayer CSFLSLayer;


  /*================================================================================
    ctor */
  CSFLSChanVeseSegmentor3D() : CSFLSSegmentor3D< TPixel >()
  {
    basicInit();
  }

  void basicInit();

  //     /* ============================================================
  //        New    */
  //     static Pointer New() 
  //     {
  //       return Pointer(new CSFLSChanVeseSegmentor3D< TPixel >);
  //     }

  double m_areaIn;
  double m_areaOut;

  double m_meanIn;
  double m_meanOut;

  /* ============================================================
   * functions
   * ============================================================*/
  void computeMeans();
  void computeMaskedMeans();
  void updateMeans();

  //void doChanVeseSegmenation();
  void doSegmenation();


  /* ============================================================
     computeForce    */
  void computeForce();

  void UpdateMeansOn() { m_bUpdateMeans = true; }
  void UpdateMeansOff() { m_bUpdateMeans = false; }
  bool UpdateMeans() { return m_bUpdateMeans; }

protected:

  bool m_bUpdateMeans;

};


#include "SFLSChanVeseSegmentor3D.txx"

#endif
