<template>
  <div>
    <UploadImages
      :max="1"
      @upload="handleImage"
      @delete="deleteImage"
    />
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
      
    }
  },
  computed: {

  },
  // created() {

  // },
  methods: {
    handleImage(files: File[]) {
      console.log(files[0]);
    },
    deleteImage() {
      console.info('deleted');
    }
  }

});
</script>
<style lang="scss">
  
</style>