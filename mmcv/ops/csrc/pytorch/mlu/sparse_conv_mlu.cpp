/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

#include <torch/script.h>
#include <vector>

template <unsigned NDim>
std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose) {
  std::cout << "GetIndicePairsForwardMLUKernelLauncher start." << std::endl;

  // The following code is copied from
  // mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu to ensure the output is
  // available for network train. The outputs of this function have correct
  // shape but wrong value.
  auto numAct = indices.size(0);
  auto kernelVolume = kernelSize[0];
  int sub_m = (int)_subM;
  int transpose = (int)_transpose;
  int batch = (int)batchSize;
  auto coorDim = indices.size(1) - 1;
//   TORCH_CHECK(NDim == coorDim, "error");
//   TORCH_CHECK(kernelSize.size() == coorDim, "error");
//   TORCH_CHECK(outSpatialShape.size() == coorDim, "error");
//   TORCH_CHECK(stride.size() == coorDim, "error");
//   TORCH_CHECK(padding.size() == coorDim, "error");
//   TORCH_CHECK(dilation.size() == coorDim, "error");

  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }

  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  torch::Tensor indicePairs = at::full({kernelVolume, 2, numAct}, -1,
                                       indices.options().dtype(at::kInt));
  torch::Tensor indiceNum = at::zeros({kernelVolume},
                            indices.options().dtype(at::kInt));
  torch::Tensor out_indices = at::zeros({numAct * kernelVolume, coorDim + 1},
                            indices.options().dtype(at::kInt));
  auto indices_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      indices, at::MemoryFormat::Contiguous);
  auto indicePairs_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      indicePairs, at::MemoryFormat::Contiguous);
  auto indiceNum_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      indiceNum, at::MemoryFormat::Contiguous);
  auto out_indices_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      out_indices, at::MemoryFormat::Contiguous);
  
  std::vector<int> input_space;
  std::vector<int> filter_space;
  std::vector<int> output_space;
  std::vector<int> padding32;
  std::vector<int> stride32;
  std::vector<int> dilation32;
  for (int i = 0; i < NDim; i++) {
      input_space.push_back(spatialShape[i]);
      filter_space.push_back(kernelSize[i]);
      output_space.push_back(outSpatialShape[i]);
      padding32.push_back(padding[i]);
      stride32.push_back(stride[i]);
      dilation32.push_back(dilation[i]);
  }
  MluOpTensorDescriptor indices_desc, out_indices_desc,
                        indicePairs_desc, indiceNum_desc;
  indices_desc.set(indices_contiguous);
  indicePairs_desc.set(indicePairs_contiguous);
  indiceNum_desc.set(indiceNum_contiguous);
  out_indices_desc.set(out_indices_contiguous);
  {
      mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
      mluOpDataType_t dtype = MLUOP_DTYPE_INT32;
      std::vector<int> dims;
      dims = {numAct, coorDim + 1};
      mluOpSetTensorDescriptor(indices_desc.desc(), layout, dtype,
                             dims.size(), dims.data());
      dims = {kernelVolume, 2, numAct};
      mluOpSetTensorDescriptor(indicePairs_desc.desc(), layout, dtype,
                             dims.size(), dims.data());
      dims = {kernelVolume};
      mluOpSetTensorDescriptor(indiceNum_desc.desc(), layout, dtype,
                             dims.size(), dims.data());
      dims = {numAct * kernelVolume, coorDim + 1};
      mluOpSetTensorDescriptor(out_indices_desc.desc(), layout, dtype,
                             dims.size(), dims.data());
  }
  
  mluOpSparseConvolutionDescriptor_t sparse_conv_desc;
  mluOpCreateSparseConvolutionDescriptor(&sparse_conv_desc);
  mluOpSetSparseConvolutionDescriptor(sparse_conv_desc, NDim + 2, batch,
  padding32.data(), stride32.data(), dilation32.data(), input_space.data(),
  filter_space.data(), output_space.data(), sub_m, transpose, 0);

  auto handle = mluOpGetCurrentHandle();
  size_t workspace_size = 0;
  mluOpGetIndicePairsWorkspaceSize(handle, sparse_conv_desc, indices_desc.desc(),
        indicePairs_desc.desc(), out_indices_desc.desc(), indiceNum_desc.desc(),
        &workspace_size);
  auto indice_workspace_size = at::empty(workspace_size,
        indices.options().dtype(at::kByte));

  auto indices_impl = torch_mlu::getMluTensorImpl(indices_contiguous);
  auto out_indices_impl = torch_mlu::getMluTensorImpl(out_indices_contiguous);
  auto indicePairs_impl = torch_mlu::getMluTensorImpl(indicePairs_contiguous);
  auto indiceNum_impl = torch_mlu::getMluTensorImpl(indiceNum_contiguous);
  auto indice_workspace_impl = torch_mlu::getMluTensorImpl(indice_workspace_size);
   
  auto indices_ptr = indices_impl->cnnlMalloc();
  auto out_indices_ptr = out_indices_impl->cnnlMalloc();
  auto indicePairs_ptr = indicePairs_impl->cnnlMalloc();
  auto indiceNum_ptr = indiceNum_impl->cnnlMalloc();
  auto indice_workspace_ptr = indice_workspace_impl->cnnlMalloc();
  
  mluOpGetIndicePairs(handle, sparse_conv_desc, indices_desc.desc(), indices_ptr,
        indice_workspace_ptr, workspace_size, indicePairs_desc.desc(),
        indicePairs_ptr, out_indices_desc.desc(), out_indices_ptr,
        indiceNum_desc.desc(), indiceNum_ptr);  
  mluOpDestroySparseConvolutionDescriptor(sparse_conv_desc);
  std::cout << "GetIndicePairsForwardMLUKernelLauncher finish." << std::endl;
  return {out_indices, indicePairs, indiceNum};  // differ from cuda code
}

torch::Tensor IndiceConvForwardMLUKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM) {
  std::cout << "IndiceConvForwardMLUKernelLauncher start." << std::endl;
  int C = filters.dim() == 4 ? filters.size(3) : filters.size(4);
  torch::Tensor output =
      at::zeros({numActOut, C}, features.options().dtype(at::kFloat));

  std::cout << "IndiceConvForwardMLUKernelLauncher finish." << std::endl;
  return output;
}

std::vector<torch::Tensor> IndiceConvBackwardMLUKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  std::cout << "IndiceConvBackwardMLUKernelLauncher start." << std::endl;
  auto indice_num_cpu = indiceNum.to({torch::kCPU});
  auto indice_num_cpu_64 = indice_num_cpu.data_ptr<int>();
  int indice_num_len = indiceNum.numel();
  int64_t indice_num[indice_num_len];
  for (int i = 0; i < indice_num_len; ++i) {
    // indice_num[i] = ((int64_t
    // *)(indice_num_cpu_64.unsafeGetTensorImpl()->data()))[i];
    indice_num[i] = (int64_t)(((int *)(indice_num_cpu_64))[i]);
    std::cout << "indice_num_cpu_64-" << i << " "
              << ((int *)(indice_num_cpu_64))[i] << std::endl;
    std::cout << "indice_num-" << i << " " << indice_num[i] << std::endl;
  }

  // generate empty input_grad
  torch::Tensor input_grad = at::zeros({features.size(0), features.size(1)},
                                       features.options().dtype(at::kFloat));
  torch::Tensor filters_grad;
  if (filters.dim() == 4) {
    int h = filters.size(0);
    int w = filters.size(1);
    int c = filters.size(2);
    int n = filters.size(3);
    filters_grad = at::zeros({h, w, c, n}, filters.options().dtype(at::kFloat));
  } else if (filters.dim() == 5) {
    int d = filters.size(0);
    int h = filters.size(1);
    int w = filters.size(2);
    int c = filters.size(3);
    int n = filters.size(4);
    filters_grad =
        at::zeros({d, h, w, c, n}, filters.options().dtype(at::kFloat));
  }

  auto features_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      features, at::MemoryFormat::Contiguous);
  auto filters_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      filters, at::MemoryFormat::Contiguous);
  auto output_grad_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      outGrad, at::MemoryFormat::Contiguous);
  auto indice_pairs_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      indicePairs, at::MemoryFormat::Contiguous);
  auto input_grad_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      features, at::MemoryFormat::Contiguous);
  auto filters_grad_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      filters, at::MemoryFormat::Contiguous);

  MluOpTensorDescriptor features_desc, output_grad_desc, filters_desc,
      indice_pairs_desc, input_grad_desc, filters_grad_desc;
  features_desc.set(features_contiguous);
  filters_desc.set(filters_contiguous);
  output_grad_desc.set(output_grad_contiguous);
  indice_pairs_desc.set(indice_pairs_contiguous);
  input_grad_desc.set(input_grad_contiguous);
  filters_grad_desc.set(filters_grad_contiguous);

  // need to set desc layout with mluOp functions
  {
    mluOpTensorLayout_t layout;
    mluOpDataType_t dtype;
    int dim;
    int dims[8];

    // features_desc
    mluOpGetTensorDescriptor(features_desc.desc(), &layout, &dtype, &dim, dims);
    mluOpSetTensorDescriptor(features_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype,
                             dim, dims);

    // filters_desc
    mluOpGetTensorDescriptor(filters_desc.desc(), &layout, &dtype, &dim, dims);
    if (dim == 4) {
      mluOpSetTensorDescriptor(filters_desc.desc(), MLUOP_LAYOUT_HWCN, dtype,
                               dim, dims);
    } else {
      mluOpSetTensorDescriptor(filters_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype,
                               dim, dims);
    }

    // output_grad_desc
    mluOpGetTensorDescriptor(output_grad_desc.desc(), &layout, &dtype, &dim,
                             dims);
    mluOpSetTensorDescriptor(output_grad_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype,
                             dim, dims);

    // indice_pairs_desc
    mluOpGetTensorDescriptor(indice_pairs_desc.desc(), &layout, &dtype, &dim,
                             dims);
    mluOpSetTensorDescriptor(indice_pairs_desc.desc(), MLUOP_LAYOUT_ARRAY,
                             dtype, dim, dims);

    // input_grad_desc
    mluOpGetTensorDescriptor(input_grad_desc.desc(), &layout, &dtype, &dim,
                             dims);
    mluOpSetTensorDescriptor(input_grad_desc.desc(), MLUOP_LAYOUT_ARRAY, dtype,
                             dim, dims);
  }

  auto handle = mluOpGetCurrentHandle();
  size_t data_workspace_size = 0;
  mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
      handle, output_grad_desc.desc(), filters_desc.desc(),
      indice_pairs_desc.desc(), input_grad_desc.desc(), indice_num, _inverse,
      &data_workspace_size);

  size_t filters_workspace_size = 0;
  mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(
      handle, features_desc.desc(), output_grad_desc.desc(),
      indice_pairs_desc.desc(), filters_grad_desc.desc(), indice_num, _inverse,
      _subM, &filters_workspace_size);

  auto indice_convbpdata_workspace =
      at::empty(data_workspace_size, features.options().dtype(at::kByte));
  auto indice_convbpfilter_workspace =
      at::empty(filters_workspace_size, filters.options().dtype(at::kByte));

  auto features_impl = torch_mlu::getMluTensorImpl(features_contiguous);
  auto filters_impl = torch_mlu::getMluTensorImpl(filters_contiguous);
  auto output_grad_impl = torch_mlu::getMluTensorImpl(output_grad_contiguous);
  auto indice_pairs_impl = torch_mlu::getMluTensorImpl(indice_pairs_contiguous);
  auto indice_convbpdata_workspace_impl =
      torch_mlu::getMluTensorImpl(indice_convbpdata_workspace);
  auto indice_convbpfilter_workspace_impl =
      torch_mlu::getMluTensorImpl(indice_convbpfilter_workspace);

  auto features_ptr = features_impl->cnnlMalloc();
  auto filters_ptr = filters_impl->cnnlMalloc();
  auto output_grad_ptr = output_grad_impl->cnnlMalloc();
  auto indice_pairs_ptr = indice_pairs_impl->cnnlMalloc();
  auto indice_convbpdata_workspace_ptr =
      indice_convbpdata_workspace_impl->cnnlMalloc();
  auto indice_convbpfilter_workspace_ptr =
      indice_convbpfilter_workspace_impl->cnnlMalloc();

  // outputs
  auto input_grad_impl = torch_mlu::getMluTensorImpl(input_grad);
  auto input_grad_ptr = input_grad_impl->cnnlMalloc();
  auto filters_grad_impl = torch_mlu::getMluTensorImpl(filters_grad);
  auto filters_grad_ptr = filters_grad_impl->cnnlMalloc();

  mluOpIndiceConvolutionBackwardData(
      handle, output_grad_desc.desc(), output_grad_ptr, filters_desc.desc(),
      filters_ptr, indice_pairs_desc.desc(), indice_pairs_ptr, indice_num,
      _inverse, _subM, indice_convbpdata_workspace_ptr, data_workspace_size,
      input_grad_desc.desc(), input_grad_ptr);

  mluOpIndiceConvolutionBackwardFilter(
      handle, features_desc.desc(), features_ptr, output_grad_desc.desc(),
      output_grad_ptr, indice_pairs_desc.desc(), indice_pairs_ptr, indice_num,
      _inverse, _subM, indice_convbpfilter_workspace_ptr,
      filters_workspace_size, filters_grad_desc.desc(), filters_grad_ptr);

  std::vector<torch::Tensor> result;
  result.push_back(input_grad);
  result.push_back(filters_grad);
  std::cout << "IndiceConvBackwardMLUKernelLauncher finish." << std::endl;
  return result;
}

torch::Tensor indice_conv_forward_mlu(torch::Tensor features,
                                      torch::Tensor filters,
                                      torch::Tensor indicePairs,
                                      torch::Tensor indiceNum,
                                      int64_t numActOut, int64_t _inverse,
                                      int64_t _subM) {
  return IndiceConvForwardMLUKernelLauncher(
      features, filters, indicePairs, indiceNum, numActOut, _inverse, _subM);
}

std::vector<torch::Tensor> indice_conv_backward_mlu(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  return IndiceConvBackwardMLUKernelLauncher(
      features, filters, outGrad, indicePairs, indiceNum, _inverse, _subM);
}

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                       torch::Tensor filters,
                                       torch::Tensor indicePairs,
                                       torch::Tensor indiceNum,
                                       int64_t numActOut, int64_t _inverse,
                                       int64_t _subM);

std::vector<torch::Tensor> indice_conv_backward_impl(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_forward_impl, MLU, indice_conv_forward_mlu);
REGISTER_DEVICE_IMPL(indice_conv_backward_impl, MLU, indice_conv_backward_mlu);

template std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher<2>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher<3>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);

template std::vector<torch::Tensor> GetIndicePairsForwardMLUKernelLauncher<4>(
    torch::Tensor indices, int64_t batchSize,
    std::vector<int64_t> outSpatialShape, std::vector<int64_t> spatialShape,
    std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    std::vector<int64_t> outPadding, int64_t _subM, int64_t _transpose);