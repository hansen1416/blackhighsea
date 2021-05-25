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