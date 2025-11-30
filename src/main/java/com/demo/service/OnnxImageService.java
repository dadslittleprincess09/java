package com.demo.service;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.springframework.stereotype.Service;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

@Service
public class OnnxImageService {

    private final OrtEnvironment env;
    private final OrtSession session;

    // model details
    private static final int MODEL_H = 224;
    private static final int MODEL_W = 224;
    private static final boolean MODEL_EXPECTS_BGR = false;

    private static final float[] MEAN = {0f, 0f, 0f};
    private static final float[] STD  = {1f, 1f, 1f};

    public OnnxImageService() throws Exception {
        env = OrtEnvironment.getEnvironment();
        URL url = getClass().getResource("/model/smartseva_multi_task.onnx");
        if (url == null) {
            throw new RuntimeException("Model file not found in resources!");
        }
        File modelFile = new File(url.toURI());
        session = env.createSession(modelFile.getAbsolutePath(), new OrtSession.SessionOptions());
    }

    //------------------------
    // IMAGE PREPROCESSING
    //------------------------
    private float[] preprocessImage(File file) throws Exception {
        BufferedImage img = ImageIO.read(file);
        if (img == null) {
            throw new RuntimeException("Unable to read image: " + file.getAbsolutePath());
        }

        Image scaled = img.getScaledInstance(MODEL_W, MODEL_H, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(MODEL_W, MODEL_H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = resized.createGraphics();
        g2.drawImage(scaled, 0, 0, null);
        g2.dispose();

        float[] input = new float[MODEL_H * MODEL_W * 3];
        int idx = 0;

        for (int y = 0; y < MODEL_H; y++) {
            for (int x = 0; x < MODEL_W; x++) {
                int pixel = resized.getRGB(x, y);
                int r = (pixel >> 16) & 255;
                int g = (pixel >> 8) & 255;
                int b = pixel & 255;

                if (MODEL_EXPECTS_BGR) {
                    input[idx++] = ((float)b / 255f - MEAN[0]) / STD[0];
                    input[idx++] = ((float)g / 255f - MEAN[1]) / STD[1];
                    input[idx++] = ((float)r / 255f - MEAN[2]) / STD[2];
                } else {
                    input[idx++] = ((float)r / 255f - MEAN[0]) / STD[0];
                    input[idx++] = ((float)g / 255f - MEAN[1]) / STD[1];
                    input[idx++] = ((float)b / 255f - MEAN[2]) / STD[2];
                }
            }
        }

        return input;
    }

    //------------------------
    // PREDICTION
    //------------------------
    public Map<String, float[]> predict(File imageFile) throws Exception {

        float[] inputData = preprocessImage(imageFile);
        long[] shape = new long[]{1, MODEL_H, MODEL_W, 3};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {

            Map<String, OnnxTensor> inputMap = new HashMap<>();

            // Get the model's actual input name
            String inputName = session.getInputNames().iterator().next();
            inputMap.put(inputName, inputTensor);

            try (Result result = session.run(inputMap)) {

                // Debug available outputs
                System.out.println("Model outputs: " + session.getOutputNames());

                float[] mainOut = extractOutput(result, "main_output");
                float[] severityOut = extractOutput(result, "severity_output");

                Map<String, float[]> outputMap = new HashMap<>();
                outputMap.put("main_output", mainOut);
                outputMap.put("severity_output", severityOut);

                return outputMap;
            }
        }
    }

    //------------------------
    // OUTPUT EXTRACTION
    //------------------------
    private float[] extractOutput(Result result, String outputName) throws Exception {

        if (!session.getOutputNames().contains(outputName)) {
            System.out.println("Output missing: " + outputName);
            return null;
        }

        OnnxValue val = result.get(outputName)
                .orElseThrow(() -> new RuntimeException(outputName + " missing"));

        Object v = val.getValue();

        if (v instanceof float[][] arr) {
            return arr[0];
        }
        if (v instanceof float[] arr) {
            return arr;
        }
        if (v instanceof double[][] darr) {
            double[] row = darr[0];
            float[] out = new float[row.length];
            for (int i = 0; i < row.length; i++) out[i] = (float) row[i];
            return out;
        }

        throw new RuntimeException("Unhandled output type: " + v.getClass());
    }

    //------------------------
    // SOFTMAX + ARGMAX HELPERS
    //------------------------
    private static float[] softmax(float[] logits) {
        if (logits == null || logits.length == 0) return logits;

        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;

        double sum = 0.0;
        double[] exps = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - max);
            sum += exps[i];
        }

        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) (exps[i] / sum);
        }

        return probs;
    }

    private static int argMax(float[] arr) {
        if (arr == null || arr.length == 0) return -1;
        int idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }
}
