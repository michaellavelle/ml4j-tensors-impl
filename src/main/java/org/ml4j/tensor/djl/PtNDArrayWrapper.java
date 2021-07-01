package org.ml4j.tensor.djl;

import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;

import java.nio.ByteBuffer;

public class PtNDArrayWrapper extends PtNDArray {

    public PtNDArrayWrapper(PtNDArray source) {
        super(source.getManager(), source.getHandle(), source.toByteBuffer());
    }
}
