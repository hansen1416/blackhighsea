<template>
  <div class='core'>
    <div class='btn b1'>
      <router-link
        :to="{name: 'stock_list'}"
      >
        <span>stocks</span>
      </router-link>
    </div>
    <div class='btn b2'>
      <router-link
        :to="{name: 'stylize'}"
      >
        <span>stylize</span>
      </router-link>
    </div>
    <div class='btn'>
    </div>
    <div class='btn'>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
// import axios, {AxiosResponse} from "axios";

export default defineComponent({
  components: {

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

  .core {
    position: absolute;
    width: 0;
    height: 0;
    top: 50%;
    left: 50%;

    .btn {
      position: absolute;
      width: 300px;
      height: 200px;
      box-sizing: border-box;
      border-radius: 6px;

      &.b1 {
        right: 0;
        bottom: 0;
        border: 1px solid #ffffff;
      }

      &.b2 {
        left: 0;
        bottom: 0;
        border: 1px solid #ffffff;
      }
    }
  }
</style>