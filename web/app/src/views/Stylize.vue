<template>
    <div>
        <div v-if="!transferedImage">
            <UploadImages :max="1" @upload="handleImage" @delete="deleteImage" />
        </div>
        <div v-if="transferedImage" class="transformed">
            <div>
                <img :src="transferedImage" />
                <a target="_blank" :href="transferedImage" download>Download</a>
            </div>
        </div>
        <section>
            <div class="theme">
                <div class="selected">
                    <img src="@/assets/img/qian.jpg" alt="" />
                </div>
            </div>
            <button class="submit" @click="submitImage">submit</button>
        </section>
        <div class="email">
            <input
                v-if="mode == 'video'"
                placeholder="email to receive the video"
                type="email"
                name="email"
                v-model="email"
            />
        </div>
        <div v-if="transferedVideo">
            <p>
                Download your video here
                <a :href="transferedVideo" download>{{ transferedVideo }}</a>
            </p>
        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import UploadImages from "@/components/UploadImages.vue";
import axios, { AxiosResponse } from "axios";

export default defineComponent({
    components: {
        UploadImages,
    },
    data() {
        return {
            origin_image: (null as unknown) as Blob,
            // ws: (null as unknown) as WebSocket,
            transferedImage: "",
            transferedVideo: "",
            email: "",
            mode: "image",
        };
    },
    created() {
        console.log("Stylize created");

        // this.email = "hansen1416@163.com";

        // this.ws = new WebSocket("ws://localhost:4601/ws/cartoongan");

        // this.ws.onopen = () => {
        //     console.log("ws Connected.");
        // };

        // this.ws.onmessage = (event: MessageEvent) => {
        //     try {
        //         const data = JSON.parse(event.data);

        //         if (data.image) {
        //             const image_name = data.image.split("/").pop();

        //             this.transferedImage = "http://localhost:4602/" + image_name;

        //             console.log(this.transferedImage);
        //         } else if (data.video) {
        //             const video_name = data.video.split("/").pop();

        //             this.transferedVideo = "http://localhost:4602/" + video_name;

        //             console.log(this.transferedVideo);
        //         }
        //     } catch (error) {
        //         console.error(error);
        //     }
        // };

        // this.ws.onclose = () => {
        //     console.log("ws Connection is closed...");
        // };

        // axios.get("http://localhost:4602/health").then((response: AxiosResponse) => {
        //     console.log(response);
        // });
    },
    computed: {},
    methods: {
        handleImage(files: File[]) {
            this.origin_image = files[0];

            const mimeType = this.origin_image.type.substring(0, 5);

            if (mimeType == "image") {
                this.mode = "image";
            } else if (mimeType == "video") {
                this.mode = "video";
            }
        },
        deleteImage() {
            console.info("deleted");
        },
        submitImage() {
            if (!this.origin_image) {
                return false;
            }

            const data = new FormData();

            data.append('img', this.origin_image)

            axios.post('http://localhost:4601/cartoongan', data)
            .then(function (response: AxiosResponse) {
                console.log(response);
            })
            .catch(function (error) {
                console.log(error);
            });

            // if (this.mode == "image") {
            //     this.ws.send("image");

            //     this.origin_image.arrayBuffer().then((buffer: ArrayBuffer) => {
            //         this.ws.send(buffer);
            //     });
            // }

            // if (this.mode == "video") {
            //     const pseudoVideo = document.createElement("video");

            //     pseudoVideo.onloadeddata = () => {
            //         if (pseudoVideo.duration >= 20) {
            //             alert("sorry video lenght must be less than 20 seconds");

            //             return;
            //         }

            //         if (!this.email) {
            //             alert(
            //                 "Please enter your email address to receive the transformed video"
            //             );

            //             return;
            //         }

            //         this.ws.send("video:" + this.email);

            //         this.origin_image.arrayBuffer().then((buffer: ArrayBuffer) => {
            //             this.ws.send(buffer);
            //         });
            //     };
            //     // create url from blob
            //     pseudoVideo.src = URL.createObjectURL(this.origin_image);
            //     pseudoVideo.load();
            // }

            return false;
        },
    },
});
</script>
<style lang="scss">
video {
    max-height: 100%;
    max-width: 100%;
}

.transformed {
    padding: 20px 30px;
    text-align: center;

    div {
        width: 300px;
        height: 300px;
        display: inline-block;
        position: relative;
        img {
            max-height: 100%;
            max-width: 100%;
        }

        a {
            text-decoration: none;
            position: absolute;
            bottom: 4px;
            right: 4px;
        }
    }
}

section {
    position: relative;
    padding: 8px 0;

    .theme {
        display: flex;
        width: 100%;
        height: 50px;
        justify-content: center;

        .selected {
            border: 4px solid #bf66e5;
        }

        img {
            max-height: 100%;
            cursor: pointer;
        }
    }

    .submit {
        position: absolute;
        right: 20px;
        bottom: 8px;
        line-height: 30px;
        cursor: pointer;
    }
}

.email {
    position: relative;
    text-align: right;
    padding: 10px 20px;
}
</style>
