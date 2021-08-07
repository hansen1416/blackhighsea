<template>
  <div class='core'>
    <Greeting/>

      <router-link
        :to="{name: 'stock_list'}"
      >
        <span>stocks</span>
      </router-link>


      <router-link
        :to="{name: 'stylize'}"
      >
        <span>stylize</span>
      </router-link>

  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
// import axios, {AxiosResponse} from "axios";
import Greeting from "@/components/Greeting.vue";

export default defineComponent({
  components: {
    Greeting,
  },
  data() {
    return {
      stockList: [],
    }
  },
  computed: {

  },
  // created() {

  // },
  methods: {

  }

});
</script>
<style lang="scss">
  #app {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    right: 0;
    background: #b92b27;  /* fallback for old browsers */
    background: -webkit-linear-gradient(to right, #1565C0, #b92b27);  /* Chrome 10-25, Safari 5.1-6 */
    background: linear-gradient(to right, #1565C0, #b92b27); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
  }
</style>