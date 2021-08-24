import "phaser";
import MainScene from "./scenes/mainScene";
import PreloadScene from "./scenes/preloadScene";
import * as handTrack from "handtrackjs";
import { mainModule } from "process";

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;

const config = {
    type: Phaser.AUTO,
    backgroundColor: "#ffffff",
    scale: {
        parent: "phaser-game",
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width: DEFAULT_WIDTH,
        height: DEFAULT_HEIGHT,
    },
    scene: [PreloadScene, MainScene],
    physics: {
        default: "arcade",
        arcade: {
            debug: false,
            gravity: { y: 400 },
        },
    },
};

type Prediction = {
    bbox: integer[];
    class: 1 | 2 | 3 | 4 | 5 | 6 | 7;
    label: "open" | "close" | "pinch" | "point" | "face" | "tip" | "pinchtip";
    score: string;
};

// later transfer to my own model
const basePath =
    "https://cdn.jsdelivr.net/npm/handtrackjs@latest/models/webmodel/";
// const basePath = "webmodel/";

const labelMap = {
    1: "open",
    2: "closed",
    3: "pinch",
    4: "point",
    5: "face",
    6: "tip",
    7: "pinchtip",
};

const defaultParams = {
    flipHorizontal: false,
    outputStride: 16,
    imageScaleFactor: 1,
    maxNumBoxes: 20,
    iouThreshold: 0.2,
    scoreThreshold: 0.6,
    modelType: "ssd320fpnlite",
    modelSize: "small",
    bboxLineWidth: "2",
    fontSize: 17,
    // basePath: basePath,
    // labelMap: labelMap,
};

const handLabel = ["open", "close", "point"];

async function main() {
    console.log("in main");

    const model = await handTrack.load(defaultParams);

    console.log("model is ready");

    const video = document.querySelector("#videoElement") as HTMLVideoElement;

    function runDetection() {
        model.detect(video).then((predictions: Prediction[]) => {
            if (predictions.length) {
                for (const pred of predictions) {
                    if (handLabel.indexOf(pred.label) != -1) {
                        console.log(
                            video.videoWidth,
                            video.videoHeight,
                            pred.bbox
                        );
                    }
                }
            }

            requestAnimationFrame(runDetection);
        });
    }

    handTrack
        .startVideo(video)
        .then((status: { status: boolean; msg: string }) => {
            if (status.status) {
                runDetection();
            }
        });
}

window.addEventListener("load", () => {
    // const game = new Phaser.Game(config)

    main();
});
