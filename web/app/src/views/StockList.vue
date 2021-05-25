<template>
  <div>
    <ul>
      <li
        v-for="stock in stockList"
        :key="stock"
      >
        <router-link
          :to="{name: 'pred_hl', params: {stock_code: stock.code}}"
        >
          <span>{{stock.code}} {{stock.name}}</span>
        </router-link>
      </li>
    </ul>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import axios, {AxiosResponse} from "axios";

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
  created() {
    axios.get(`${process.env.VUE_APP_PY_API}/stocklist`)
    .then((response: AxiosResponse) => {
      if(response.status == 200) {
        const data: Record<string, string> = response.data;

        if (data) {

          this.$store.commit('setStocknames', data);

          for (const prop in data) {

            this.stockList.push({code: prop, name: data[prop]} as never)
          }
        }
      }
    })
    .catch((e) => {
      console.info(e);
    });
  },
  methods: {

  }

});
</script>
