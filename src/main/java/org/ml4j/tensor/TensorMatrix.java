package org.ml4j.tensor;

import org.jvmpy.symbolictensors.Size;
import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;

public class TensorMatrix<T extends Tensor<T, ?>> implements Matrix, EditableMatrix, InterrimMatrix {

    private T tensor;
    private boolean immutable;

    public TensorMatrix(T tensor) {
        this.tensor = tensor;
    }

    protected TensorMatrix create(T tensor) {
        return new TensorMatrix(tensor);
    }

    protected T extract(Matrix matrix) {
        if (matrix instanceof TensorMatrix) {
            return ((TensorMatrix<T>)matrix).tensor;
        } else {
            throw new IllegalArgumentException();
        }
    }

    @Override
    public void putRow(int i, Matrix matrix) {
        tensor.putTensor(extract(matrix), i, -1);
    }

    @Override
    public void putColumn(int i, Matrix matrix) {
        tensor.putTensor(extract(matrix), -1, i);
    }

    @Override
    public void put(int i, int i1, float v) {
        tensor.put(v, new int[] {i, i1});
    }

    @Override
    public void put(int i, float v) {
        tensor.put(i, v);
    }

    @Override
    public EditableMatrix subi(float v) {
        return create(tensor.sub_(v));
    }

    @Override
    public EditableMatrix divi(float v) {
        return create(tensor.div_(v));
    }

    @Override
    public EditableMatrix addi(float v) {
        return create(tensor.add_(v));
    }

    @Override
    public EditableMatrix muli(float v) {
        return create(tensor.mul_(v));
    }

    @Override
    public EditableMatrix subiColumnVector(Matrix matrix) {
        tensor.sub_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix subiRowVector(Matrix matrix) {
        tensor.sub_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix diviColumnVector(Matrix matrix) {
        tensor.div_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix diviRowVector(Matrix matrix) {
        tensor.div_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix addiRowVector(Matrix matrix) {
        tensor.add_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix addiColumnVector(Matrix matrix) {
        tensor.add_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix muliColumnVector(Matrix matrix) {
        tensor.mul_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix muliRowVector(Matrix matrix) {
        tensor.mul_(extract(matrix));
        return this;
    }

    @Override
    public EditableMatrix muli(Matrix matrix) {
        return create(tensor.mul_(extract(matrix)));
    }

    @Override
    public EditableMatrix addi(Matrix matrix) {
        return create(tensor.add_(extract(matrix)));
    }

    @Override
    public EditableMatrix divi(Matrix matrix) {
        return create(tensor.div_(extract(matrix)));
    }

    @Override
    public EditableMatrix subi(Matrix matrix) {
        return create(tensor.sub_(extract(matrix)));
    }

    @Override
    public EditableMatrix expi() {
        this.tensor = tensor.exp();
        return this;
    }


    @Override
    public void reshape(int i, int i1) {
        tensor.resize_(new Size(i, i1));
    }

    @Override
    public int getColumns() {
        return tensor.size(1);
    }

    @Override
    public int getRows() {
        return tensor.size(0);
    }

    @Override
    public float[] getRowByRowArray() {
        return tensor.getDataAsFloatArray();
    }

    @Override
    public float[] getColumnByColumnArray() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float[] toColumnByColumnArray() {
        return tensor.t().getDataAsFloatArray();
    }

    @Override
    public Matrix getColumn(int i) {
        Matrix mat =  create(tensor.getTensor(-1, i));

        return mat;
        
    }

    @Override
    public Matrix getRow(int i) {
        Matrix mat =  create(tensor.getTensor(i, -1));
        return mat;
    }

    @Override
    public Matrix appendVertically(Matrix matrix) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix get(int[] ints, int[] ints1) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix transpose() {
        return create(tensor.t());
    }

    @Override
    public Matrix columnSums() {
        return create(tensor.sum(0));
    }

    @Override
    public Matrix rowSums() {
        return create(tensor.sum(1));
    }

    @Override
    public int[] columnArgmaxs() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix mulColumnVector(Matrix matrix) {
        return create(tensor.mul(extract(matrix)));
    }

    @Override
    public Matrix mulRowVector(Matrix matrix) {
        return create(tensor.mul(extract(matrix)));
    }

    @Override
    public Matrix addColumnVector(Matrix matrix) {
        return create(tensor.add(extract(matrix)));
    }

    @Override
    public Matrix addRowVector(Matrix matrix) {
        return create(tensor.add(extract(matrix)));
    }

    @Override
    public Matrix divColumnVector(Matrix matrix) {
        return create(tensor.div(extract(matrix)));
    }

    @Override
    public Matrix divRowVector(Matrix matrix) {
        return create(tensor.div(extract(matrix)));
    }

    @Override
    public Matrix subColumnVector(Matrix matrix) {
        return create(tensor.sub(extract(matrix)));
    }

    @Override
    public Matrix subRowVector(Matrix matrix) {
        return create(tensor.sub(extract(matrix)));
    }

    @Override
    public Matrix mmul(Matrix matrix) {
        return create(tensor.matmul(extract(matrix)));
    }

    @Override
    public Matrix softDup() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float get(int row, int col) {
        return tensor.get(row * getColumns() + col);
    }

    @Override
    public Matrix appendHorizontally(Matrix matrix) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix getColumns(int[] ints) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Matrix getRows(int[] ints) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int argmax() {
        int argmax = (int)tensor.argMax().getDataAsFloatArray()[0];
        return argmax;
    }

    @Override
    public float get(int i) {
        return tensor.get(i);
    }

    @Override
    public float sum() {
        return tensor.sum().getDataAsFloatArray()[0];
    }

    @Override
    public int getLength() {
        return tensor.numel();
    }

    @Override
    public Matrix mul(float v) {
        return create(tensor.mul(v));
    }

    @Override
    public Matrix add(float v) {
        return create(tensor.add(v));
    }

    @Override
    public Matrix div(float v) {
        return create(tensor.div(v));
    }

    @Override
    public Matrix sub(float v) {
        return create(tensor.sub(v));
    }

    @Override
    public Matrix div(Matrix matrix) {
        return create(tensor.div(extract(matrix)));
    }

    @Override
    public Matrix sub(Matrix matrix) {
        return create(tensor.sub(extract(matrix)));
    }

    @Override
    public Matrix mul(Matrix matrix) {
        return create(tensor.mul(extract(matrix)));
    }

    @Override
    public Matrix add(Matrix matrix) {
        return create(tensor.add(extract(matrix)));
    }

    @Override
    public Matrix sigmoid() {
        return create(tensor.sigmoid());
    }

    @Override
    public Matrix log() {
        // TODO
        return create(tensor.log());
    }

    @Override
    public EditableMatrix logi() {
        // TODO
        this.tensor = tensor.log();
        return this;
    }

    @Override
    public void close() {
        // No-op
    }

    @Override
    public boolean isClosed() {
        return false;
    }

    @Override
    public Matrix dup() {
        return create(tensor.cloneTensor());
    }

    @Override
    public EditableMatrix asEditableMatrix() {
        return this;
    }

    @Override
    public InterrimMatrix asInterrimMatrix() {
        return this;
    }

    @Override
    public boolean isImmutable() {
        return immutable;
    }

    @Override
    public void setImmutable(boolean immutable) {
        this.immutable = immutable;
    }
}
