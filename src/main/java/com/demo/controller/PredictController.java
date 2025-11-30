package com.demo.controller;

import java.io.File;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.demo.service.OnnxImageService;

@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api")
public class PredictController {

    @Autowired
    private OnnxImageService onnxImageService;

    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestParam("file") MultipartFile file) {
        try {

            if (file.isEmpty()) {
                return ResponseEntity.badRequest().body("File is empty!");
            }

            // Save file temporarily
            File conv = File.createTempFile("upload_", "_" + file.getOriginalFilename());
            file.transferTo(conv);

            // Run prediction
            Map<String, float[]> prediction = onnxImageService.predict(conv);

            // Delete temp file
            conv.delete();

            // Return response
            return ResponseEntity.ok(Map.of(
                    "main_output", prediction.get("main_output"),
                    "severity_output", prediction.get("severity_output")
            ));

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error: " + e.getMessage());
        }
    }
}
