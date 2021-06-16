package org.ml4j.tensor;
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

import com.google.common.base.Supplier;
import org.jvmpy.symbolictensors.Size;
import org.jvmpy.symbolictensors.TensorDataContainer;
import org.ml4j.autograd.arithmetic.operations.ArithmeticOperations;

public interface TensorOperations<T> extends Supplier<T>, TensorDataContainer, java.util.function.Supplier<T>, ArithmeticOperations<T> {

    int numel();

    T sum(int... dims);

    float get(int index);

    float get(int...indexes);

    T getTensor(int...indexes);

    void putTensor(T tensor, int...indexes);

    T argMax(int axis);

    T argMax();

    void put(int index, float value);

    void put(float value, int...indexes);

    T mean();

    T norm();

    T mul_(T other);

    T add_(float v);

    T div_(float v);

    T sub_(float v);

    T div_(T other);

    T mul_(float v);

    T columnSums();

    T rowSums();

    T cloneTensor();

    T matmul(T other);

    T t();

    Size size();

    int size(int dim);

    T size_(Size size);

    T zero_();

    T normal_(float v1, float v2);

    T fill_(float value);

    T view(Size size);

    T reshape_(Size size);

    T view(int... dims);

    T relu();

    T bernoulli();

    T sigmoid();

    T exp();

    T log();

    void close();
}