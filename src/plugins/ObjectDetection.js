import * as tf from "@tensorflow/tfjs";

tf.setBackend("webgl");

export default class ObjectDetection {

    constructor({
        model_url,
        classes_url,
    }) {
        this.model = null
        this.classes = null

        this.model_url = model_url;
        this.classes_url = classes_url;
    }

    isModelLoaded = () => {
        return this.model !== null
    }
    isClassesLoaded = () => {
        return this.classes !== null
    }

    loadModel = async () => {
        try {
            this.model = await tf.loadGraphModel(this.model_url);
            console.log('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model:', error);
        }
    }


    loadClasses = async () => {
        try {
            const response = await fetch(this.classes_url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.classes = await response.json();
            console.log('Classes loaded successfully');
        } catch (error) {
            console.error('Error loading classes:', error);
        }
    }

    detect = async (image, detection_threshold = 0.5) => {
        if (!this.isModelLoaded()) {
            console.error('Model not yet loaded');
            return undefined;
        }

        if (!this.isClassesLoaded()) {
            console.error('Classes not yet loaded');
            return undefined;
        }

        const detection = await this.model.executeAsync(
            await this._preprocessImg(image)
        );

        const detectionObjects = this._buildDetectedObjects({
            classes: detection[0].dataSync(),
            scores: detection[1].arraySync(),
            boxes: detection[5].arraySync(),
            classesDir: this.classes,
            threshold: detection_threshold,
            frame: image
        });

        return detectionObjects
    }

    _preprocessImg = async (image) => {
        const tfimg = tf.browser.fromPixels(image).toInt();
        const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
        return expandedimg;
    }

    _buildDetectedObjects = ({ scores, threshold, boxes, classes, classesDir, frame }) => {
        const detectionObjects = [];

        // Check if scores is an array and has data
        if (Array.isArray(scores) && scores.length > 0) {
            // Assuming scores is a 2D array with scores in the first row
            const scoreArray = scores[0];

            // Iterate over the scores
            scoreArray.forEach((score, i) => {
                if (score > threshold) {
                    const bbox = [];
                    const minY = boxes[0][i][0] * frame.naturalHeight;
                    const minX = boxes[0][i][1] * frame.naturalWidth;
                    const maxY = boxes[0][i][2] * frame.naturalHeight;
                    const maxX = boxes[0][i][3] * frame.naturalWidth;

                    bbox[0] = minX;
                    bbox[1] = minY;
                    bbox[2] = maxX - minX;
                    bbox[3] = maxY - minY;

                    detectionObjects.push({
                        class: classesDir[classes[i]],
                        score: score.toFixed(4),
                        bbox: bbox,
                    });
                }
            });
        } else {
            console.error('Invalid scores format:', scores);
        }

        return detectionObjects;
    }

}