/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package org.ml4j.tensor.dl4j;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.tensor.Tensor;
import org.ml4j.tensor.TensorOperations;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An AutogradValue implementation that supports the operations defined by TensorOperations,
 * and that takes advantage of the fact that the wrapped data also implements TensorOperations
 * by implementing default DifferentiableWrappedTensorOperations methods.
 *
 * @author Michael Lavelle
 */
public interface DL4JTensor extends AutogradValue<DL4JTensor, DL4JTensorOperations, Size>, TensorOperations<DL4JTensor>, org.ml4j.autograd.DataSupplier<DL4JTensorOperations>, Tensor<DL4JTensor, DL4JTensorOperations> {
    INDArray getNDArray();
}
