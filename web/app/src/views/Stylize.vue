<template>
  <div>
    <UploadImages
      :max="1"
      @upload="handleImage"
      @delete="deleteImage"
    />
    <button
      @click="submitImage"
    >submit</button>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import axios, {AxiosResponse} from "axios";
import UploadImages from "@/components/UploadImages.vue";
// import { io } from "socket.io-client";


export default defineComponent({
  components: {
    UploadImages,
  },
  data() {
    return {
      origin_image: null as unknown as File,
    }
  },
  computed: {

  },
  // created() {

  // },
  sockets: {
        connect: function () {
            console.log('socket connected')
        },
        customEmit: function (data: any) {
            console.log('this method was fired by the socket server. eg: io.emit("customEmit", data)')
        }
    },
  methods: {
    handleImage(files: File[]) {
      this.origin_image = files[0];
    },
    deleteImage() {
      console.info('deleted');
    },
    submitImage() {

      const data = new FormData();

      data.append('origin_image', this.origin_image);

      axios.post('http://localhost:4601/stylize', data)
      .then(function (response: AxiosResponse){
        console.log(response);
      })
      .catch(function (error: any) {
        console.log(error);
      });
    }
  }

});
</script>
<style lang="scss">
  
</style>