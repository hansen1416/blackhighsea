<template>
    <div>
        <div>
            <UploadImages :max="1" @upload="handleImage" @delete="deleteImage" />
            <button @click="submitImage">submit</button>
        </div>
        <div>
            <img 
                v-if ="transferedImage"
                :src="transferedImage" 
            />
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
            transferedImage: '',
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
                
                const image_name = event.data.split('/').pop();

                this.transferedImage = 'http://localhost:4602/' + image_name;

                console.log(this.transferedImage)

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
            this.origin_image.arrayBuffer().then((buffer) => {
                this.ws.send(buffer);
            });

            return false;

            // console.log(this.origin_image instanceof Blob);

            // const reader = new FileReader();
            // let rawData = new ArrayBuffer();

            // reader.onload = function(evt) {
            //     rawData = evt.target.result;

            //     console.log(rawData);

            //     // ws.send(rawData);
            // }

            // reader.readAsArrayBuffer(this.origin_image);

            // const data = new FormData();

            // this.ws.send({file: this.origin_image});

            // data.append("origin_image", this.origin_image);

            // axios
            //     .post("http://localhost:4601/stylize", data)
            //     .then(function (response: AxiosResponse) {
            //         console.log(response);
            //     })
            //     .catch(function (error: any) {
            //         console.log(error);
            //     });
        },
    },
});
</script>
<style lang="scss"></style>
