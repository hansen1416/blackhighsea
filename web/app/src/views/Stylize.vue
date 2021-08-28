<template>
    <div>
        <div>
            <UploadImages :max="1" @upload="handleImage" @delete="deleteImage" />
            <button @click="submitImage">submit</button>
        </div>
        <div>
            <img v-if="transferedImage" :src="transferedImage" />
            <video v-if="transferedVideo" :src="transferedVideo" width="" height="" controls></video>
        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
// import axios, { AxiosResponse } from "axios";
import UploadImages from "@/components/UploadImages.vue";

export default defineComponent({
    components: {
        UploadImages,
    },
    data() {
        return {
            origin_image: (null as unknown) as Blob,
            ws: (null as unknown) as WebSocket,
            transferedImage: "",
            transferedVideo: "",
        };
    },
    created() {
        console.log("Stylize created");

        this.ws = new WebSocket("ws://127.0.0.1:4601/ws/cartoongan");

        this.ws.onopen = () => {
            console.log("ws Connected.");
        };

        this.ws.onmessage = (event: MessageEvent) => {
            try {
                const data = JSON.parse(event.data);

                if (data.image) {
                    const image_name = data.image.split("/").pop();

                    this.transferedImage = "http://localhost:4602/" + image_name;

                    console.log(this.transferedImage);
                }else if (data.video) {
                    const video_name = data.video.split("/").pop();

                    this.transferedVideo = "http://localhost:4602/" + video_name;

                    console.log(this.transferedVideo);
                }
            } catch (error) {
                console.error(error);
            }
        };

        this.ws.onclose = () => {
            console.log("ws Connection is closed...");
        };
    },
    computed: {},
    methods: {
        handleImage(files: File[]) {
            this.origin_image = files[0];
        },
        deleteImage() {
            console.info("deleted");
        },
        submitImage() {
            if (!this.origin_image) {
                return false;
            }

            if (this.origin_image.type.substring(0, 5) == "image") {
                this.ws.send("image");

                this.origin_image.arrayBuffer().then((buffer: ArrayBuffer) => {
                    this.ws.send(buffer);
                });
            }

            if (this.origin_image.type.substring(0, 5) == "video") {
                this.ws.send("video");

                this.origin_image.arrayBuffer().then((buffer: ArrayBuffer) => {
                    this.ws.send(buffer);
                });
            }

            return false;
        },
    },
});
</script>
<style lang="scss"></style>
