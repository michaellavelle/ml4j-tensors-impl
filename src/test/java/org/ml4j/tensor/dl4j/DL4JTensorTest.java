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

import ai.djl.ndarray.types.Shape;
import org.junit.Assert;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.autograd.impl.AutogradValueProperties;
import org.ml4j.tensor.TensorTestBase;
import org.ml4j.tensor.djl.DJLTensorOperations;
import org.ml4j.tensor.ml4j.ML4JTensorOperations;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JTensorTest extends TensorTestBase<DL4JTensor, DL4JTensorOperations> {

	/*
	private DJLTensorImpl createFromML4JTensor(ML4JTensorImpl t) {
		NDArray ndArray = DJLTensorFactory.getManager().create(t.getDataAsFloatArray(),
				DJLTensorFactory.getShape(t.size()));
		DJLTensorOperations ops = new DJLTensorOperationsImpl(ndArray);
		return new DJLTensorImpl(() -> ops, t.size(), t.requires_grad(), false);
	}

	 */

	/*
	@Test
	public void switchTest() {

		var a = createGradValue(-4f, true, new Size(2, 2)).name_("a");

		var b = createGradValue(-4f, true, new Size(2, 2)).name_("a");

		if (!isNativeGradientExpected()) {
			a.getGradNode().setDisableNativeGradient(true);
		}

		var c = a.add(b);


		DJLTensorImpl t = (DJLTensorImpl)c;

		//t.getGradNode().setDisableNativeGradient(true);


		//ML4JTensor s = new ML4JTensor(t, ML4JTensorFactory.DEFAULT_DIRECTED_COMPONENTS_CONTEXT);

		ML4JTensor s = new ML4JTensorWrapperImpl(ML4JTensorFactory.DEFAULT_DIRECTED_COMPONENTS_CONTEXT, t);

		var u = s.mul(s);
		System.out.println("CL:" + u.getClass());
		assertEquals(createData(-8f, new Size(2, 2)), c.data().get());

		u.backward();

		assertEquals(createData(-16f, new Size(2, 2)), t.grad().data().get());


		assertEquals(createData(-16f, new Size(2, 2)), a.grad().data().get());

		if (isNativeGradientSupported()) {
			Assert.assertEquals(isNativeGradientExpected(), a.grad().isNativeGradient());
		}
	}


	@Test
	public void switchTest2() {

		var a = createGradValue(-4f, true, new Size(2, 2)).name_("a");

		var b = createGradValue(-4f, true, new Size(2, 2)).name_("a");

		if (!isNativeGradientExpected()) {
			a.getGradNode().setDisableNativeGradient(true);
		}

		var c = a.add(b);


		DJLTensorImpl t = (DJLTensorImpl)c;

		ML4JTensor s = new ML4JTensorWrapperImpl(ML4JTensorFactory.DEFAULT_DIRECTED_COMPONENTS_CONTEXT, t);

		DifferentiableUnaryOperator<ML4JTensor, ML4JTensorOperations, Size> differentiableUnaryOperator = new
				DifferentiableUnaryOperator<ML4JTensor, ML4JTensorOperations, Size>() {

					@Override
					public UnaryOperator<ML4JTensorOperations> getForward() {
						return f -> f.mul(2);
					}

					@Override
					public BiFunction<ML4JTensor, ML4JTensor, ML4JTensor> getBackwardThis() {
						return (g, p) -> {

							System.out.println("BBBBBBBBB:" + p.getClass());
							System.out.println("BBBBBBBBC:" + g.getClass());

							return g.mul(2); };
					}

					@Override
					public UnaryOperator<Size> getContextMapper() {
						return s -> s;
					}
				};
		var u = s.apply(differentiableUnaryOperator).requires_grad_(true);
		assertEquals(createData(-8f, new Size(2, 2)), c.data().get());


		u.backward();

		assertEquals(createData(2f, new Size(2, 2)), t.grad().data().get());

		assertEquals(createData(2f, new Size(2, 2)), a.grad().data().get());

		Assert.assertFalse(a.grad().isNativeGradient());
	}


	 */

	@Override
	protected DL4JTensor createGradValue(float value, boolean requires_grad) {
        return new DL4JTensorImpl(() -> createData(value), new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad)).requires_grad_(requires_grad);
	}

	@Override
	protected DL4JTensor createGradValue(DL4JTensorOperations value, boolean requires_grad) {
        return new DL4JTensorImpl(() -> value, new AutogradValueProperties<Size>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad)).requires_grad_(requires_grad);
	}

	@Override
	protected DL4JTensor createGradValue(float value, boolean requires_grad, Size size) {
		return new DL4JTensorImpl(() -> createData(value, size), new AutogradValueProperties<Size>().setRegistry(registry).setContext(size).setRequires_grad(requires_grad)).requires_grad_(requires_grad);
	}

	@Override
	protected DL4JTensorOperations createData(float value) {
		return new DL4JTensorOperationsImpl(size, value);
	}

	@Override
	protected DL4JTensorOperations createData(float value, Size size) {
		return new DL4JTensorOperationsImpl(size, value);
	}

	@Override
	protected void assertEquals(DL4JTensorOperations value1, DL4JTensorOperations value2) {
		float[] m1 = value1.getDataAsFloatArray();
		float[] m2 = value2.getDataAsFloatArray();
		Assert.assertEquals(m1.length,  m2.length);
		for (int i = 0; i < m1.length; i++) {

			Assert.assertEquals(m1[i], m2[i], 0.01f);
		}
	}

	protected void assertEquals(DJLTensorOperations value1, ML4JTensorOperations value2) {
		float[] m1 = value1.getNDArray().toFloatArray();
		float[] m2 = value2.getMatrix().getRowByRowArray();
		Assert.assertEquals(m1.length,  m2.length);
		for (int i = 0; i < m1.length; i++) {

			Assert.assertEquals(m1[i], m2[i], 0.01f);
		}
	}


	@Override
	protected DL4JTensorOperations add(DL4JTensorOperations value1, DL4JTensorOperations value2) {
		return value1.add(value2);
	}

	@Override
	protected DL4JTensorOperations mul(DL4JTensorOperations value1, float value2) {
		return value1.mul(value2);
	}

	@Override
	protected boolean isNativeGradientSupported() {
		return false;
	}

	@Override
	protected boolean isNativeGradientExpected() {
		return false;
	}

	@Override
	protected void assertSize(DL4JTensor tensor, Size s) {
		Assert.assertEquals(tensor.size().dimensions().length, s.dimensions().length);
		for (int i = 0; i < s.dimensions().length; i++) {
			Assert.assertEquals(tensor.size().dimensions()[i], s.dimensions()[i]);

		}
		Assert.assertEquals(tensor.getNDArray().shape().length, s.dimensions().length);
		for (int i = 0; i < s.dimensions().length; i++) {
			Assert.assertEquals(tensor.getNDArray().shape()[i], s.dimensions()[i]);

		}
	}

	@Override
	protected DL4JTensor createGradValue(float[] data, int... dims) {
		return new DL4JTensorImpl(Nd4j.create(data, dims), false, registry);
	}


	@Override
	public void testMatMul() {
	}

	@Override
	public void test_get_row() {
		//super.test_get_row();
	}

	@Override
	public void test_sum() {
		//super.test_sum();
	}

	@Override
	public void test_example() {

	}
}
