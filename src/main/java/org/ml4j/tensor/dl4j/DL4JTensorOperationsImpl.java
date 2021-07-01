package org.ml4j.tensor.dl4j;

import org.jvmpy.symbolictensors.Operation;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.djl.DJLTensorOperations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

public class DL4JTensorOperationsImpl implements DL4JTensorOperations {

	private INDArray ndArray;
	private int[] shape;

	public DL4JTensorOperationsImpl(INDArray a) {
		this.ndArray = a;
		this.shape = getShape(a.shape()).dimensions();
	}

	public DL4JTensorOperationsImpl(DJLTensorOperations other) {
		this.ndArray = Nd4j.create(other.getDataAsFloatArray(), other.size().dimensions());
		this.shape = other.size().dimensions();
	}

	protected DL4JTensorOperationsImpl(Size a) {
		this.ndArray = null;
		this.shape = a.dimensions();
	}

	/*
	// @Override
	public float item() {
		float[] data = getDataAsFloatArray();
		if (data.length != 1) {
			throw new IllegalStateException();
		}
		return data[0];
	}

	 */

	@Override
	public String toString() {
		return "" + getNDArray();
	}

	public Size getShape() {
		return new Size(shape);
	}

	public DL4JTensorOperationsImpl(Size size, float initialValue) {
		if (size.dimensions().length > 0) {
			this.ndArray = Nd4j.ones(size.dimensions()).mul(initialValue);
		} else {
			this.ndArray = Nd4j.scalar(initialValue);
		}
	}

	public INDArray getNDArray() {
		return ndArray;
	}

	public DL4JTensorOperations create(INDArray other) {
		return new DL4JTensorOperationsImpl(other);
	}

	public final Supplier<DL4JTensorOperations> zero(Size shape) {
		return () -> create(Nd4j.create(shape.dimensions()));
	}

	public final Supplier<DL4JTensorOperations> one(Size shape) {
		return () -> create(Nd4j.ones(shape.dimensions()));
	}

	@Override
	public DL4JTensorOperations mul(DL4JTensorOperations other) {
		return applyBinaryOperation(other, (f, s) -> f.mul(s));
	}

	@Override
	public DL4JTensorOperations t() {
		return create(getNDArray().transpose());
	}

	@Override
	public DL4JTensorOperations mul(float other) {
		return applyWithFloatOperation(other, (f, s) -> f.mul(s));
	}

	@Override
	public DL4JTensorOperations add(DL4JTensorOperations other) {
		return applyBinaryOperation(other, (f, s) -> f.add(s));
	}

	protected DL4JTensorOperations applyBinaryOperation(DL4JTensorOperations other, BinaryOperator<INDArray> op) {
		return create(op.apply(getNDArray(), other.getNDArray()));
	}

	protected DL4JTensorOperations applyUnaryOperation(UnaryOperator<INDArray> op) {
		return create(op.apply(getNDArray()));
	}

	protected DL4JTensorOperations applyWithFloatOperation(float other, BiFunction<INDArray, Float, INDArray> op) {
		return create(op.apply(getNDArray(), other));
	}

	@Override
	public DL4JTensorOperations add(float other) {
		return applyWithFloatOperation(other, (f, s) -> f.add(s));
	}

	@Override
	public DL4JTensorOperations sub(float other) {
		return applyWithFloatOperation(other, (f, s) -> f.sub(s));
	}

	@Override
	public DL4JTensorOperations div(float other) {
		return applyWithFloatOperation(other, (f, s) -> f.div(s));
	}

	@Override
	public DL4JTensorOperations relu() {
		return applyUnaryOperation(n -> n.gt(Nd4j.zeros(getNDArray().shape())).mul(getNDArray()));
	}

	@Override
	public DL4JTensorOperations bernoulli() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations sigmoid() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations exp() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations log() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void close() {

	}

	@Override
	public boolean isNativeGradient() {
		return false;
	}

	@Override
	public DL4JTensorOperations sum(int... dims) {
		if (dims.length > 0) {
			return create(getNDArray().sum(dims));
		} else {
			if (getNDArray().shape().length == 0) {
				return new DL4JTensorOperationsImpl(new Size(), getNDArray().getFloat(0));
			}
			return create(getNDArray().sum(0));
		}
	}

	@Override
	public float get(int index) {
		return getNDArray().getFloat(index);
	}

	@Override
	public float get(int... indexes) {
		return getNDArray().getFloat(indexes);
	}

	@Override
	public DL4JTensorOperations getTensor(int... indexes) {
		List<String> inds = new ArrayList<>();
		for (int i = 0; i < indexes.length; i++) {
			inds.add(indexes[i] == -1 ? ":" : (indexes[i] + ""));
		}
		String r = inds.toString().replace("[","").replace("]", "");
		if (indexes.length == 2 && getNDArray().shape().length == 1) {
		//return create(getNDArray().reshape(1, getShape().dimensions()[0]).get(new NDIndex(r)), false);
		} else {
			//return create(getNDArray().get(new NDIndex(r)), false);
		}
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations getTensor(int[]... ranges) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void putTensor(DL4JTensorOperations tensor, int[]... indexes) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void putTensor(DL4JTensorOperations tensor, int... indexes) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations argMax(int axis) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations argMax() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void put(int index, float value) {
		throw new UnsupportedOperationException();

	}

	@Override
	public void put(float value, int... indexes) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations mean() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations norm() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations mul_(DL4JTensorOperations other) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations add_(float v) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations div_(float v) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations sub_(float v) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations div_(DL4JTensorOperations other) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations mul_(float v) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations columnSums() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations rowSums() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations cloneTensor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations gt(float value) {
		return create(getNDArray().gt(value));
	}

	@Override
	public DL4JTensorOperations gte(float value) {
		return create(getNDArray().gte(value));
	}

	@Override
	public DL4JTensorOperations add_(DL4JTensorOperations other) {
		getNDArray().addi(other.getNDArray());
		return this;
	}

	@Override
	public DL4JTensorOperations sub_(DL4JTensorOperations other) {
		getNDArray().subi(other.getNDArray());
		return this;
	}

	@Override
	public DL4JTensorOperations neg() {
		return applyUnaryOperation(n -> n.neg());
	}

	@Override
	public DL4JTensorOperations sub(DL4JTensorOperations other) {
		return applyBinaryOperation(other, (f, s) -> f.sub(s));
	}

	@Override
	public DL4JTensorOperations div(DL4JTensorOperations other) {
		return applyBinaryOperation(other, (f, s) -> f.div(s));
	}

	@Override
	public DL4JTensorOperations matmul(DL4JTensorOperations other) {
		DL4JTensorOperations ret = applyBinaryOperation(other, (f, s) -> f.mmul(s));
		return ret;
	}

	@Override
	public int numel() {
		return size().numel();
	}
	/*
	@Override
	public DL4JTensorOperations filter(Range... ranges) {

		int[] dims = new int[ranges.length];
		int ind = 0;
		int zeroCount = 0;
		for (Range r : ranges) {
			dims[ind] = r.getSize((int) size().decompose().get(ind).numel());
			if (dims[ind] == 0) {
				zeroCount++;
			}
			ind++;
		}
		long[] dims2 = new long[ranges.length - zeroCount];
		int ind2 = 0;
		for (int i = 0; i < dims.length; i++) {
			if (dims[i] != 0) {
				dims2[ind2] = dims[i];
				ind2++;
			}
		}

		// TODO
		// TODO Auto-generated method stub

		//Size newSize = new Size(dims2);
		// TODO
		for (long d : dims2) {
			System.out.println("DIM:" + d);
		}
		for (long d : ndArray.shape()) {
			System.out.println("SHAPE:" + d);
		}
		boolean same = true;
		if (dims2.length == ndArray.shape().length) {
			for (int i = 0; i < dims2.length; i++) {
				if (dims2[i] != ndArray.shape()[i]) {
					same = false;
				}
			}
		}
		return new DL4JTensorOperations(ndArray);

	}

	 */

	@Override
	public DL4JTensorOperations view(int... ints) {
		if (new Size(ints).numel() != numel()) {
			throw new IllegalArgumentException("Number of elements do not match");
		}
		if (numel() != getNDArray().length()) {
			throw new IllegalArgumentException("Number of elements do not match");
		}
		return applyUnaryOperation(t -> t.reshape(ints));
	}

	/*
	//@Override
	public DL4JTensorOperations transpose(int... ints) {
		return applyUnaryOperation(t -> t.swapAxes(ints[0], ints[1]));
	}

	 */

	@Override
	public DL4JTensorOperations view(Size size) {
		if (size.numel() != numel()) {
			throw new IllegalArgumentException("Number of elements do not match");
		}
		return applyUnaryOperation(t -> t.reshape(size.dimensions()));
	}

	@Override
	public DL4JTensorOperations reshape(Size size) {
		return applyUnaryOperation(t -> t.reshape(size.dimensions()));
	}

	@Override
	public DL4JTensorOperations resize_(Size size) {
		this.ndArray = getNDArray().reshape(size.dimensions());
		return this;
	}
	/*
	@Override
	public DL4JTensorOperations dup() {
		return new DL4JTensorOperationsImpl(ndArray.dup());
	}

	 */

	@Override
	public int size(int d) {
		if (d < 0) d = size().dimensions().length + d;
		return size().dimensions()[d];
	}

	/*
	@Override
	public DL4JTensorOperations size_(Size size) {
		throw new UnsupportedOperationException();
	}

	 */

	@Override
	public DL4JTensorOperations zero_() {
		this.ndArray = Nd4j.zeros();
		return this;
	}

	@Override
	public DL4JTensorOperations normal_(float v1, float v2) {
		this.ndArray = Nd4j.zeros();
		return this;
	}

	@Override
	public DL4JTensorOperations fill_(float value) {
		throw new UnsupportedOperationException();
	}

	@Override
	public float[] getDataAsFloatArray() {
		return this.getNDArray().data().asFloat();
	}

	@Override
	public Size size() {
		return getShape(ndArray.shape());
	}

	public static Size getShape(long[] dims) {
		int[] dims2 = new int[dims.length];
		for (int i = 0; i < dims.length; i++) {
			dims2[i] = (int) dims[i];
		}
		return new Size(dims2);
	}



	@Override
	public void performInlineOperation(Operation<DL4JTensorOperations, Size> operation) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations performUnaryMappingOperation(Operation<DL4JTensorOperations, Size> operation) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DL4JTensorOperations get() {
		return this;
	}
}
