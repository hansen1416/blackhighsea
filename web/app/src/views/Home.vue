<template>
  <div>
    <ul>
      <li
        v-for="code in allStocks"
        :key="code"
      >
        <router-link
          :to="{name: 'pred_hl', params: {stock_code: code}}"
        >
          <span>{{code}}</span>
        </router-link>
      </li>
    </ul>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';

export default defineComponent({
  components: {

  },
  computed: {
    allStocks(): string[] {
      return ['600104', '600198'];
    }
  },
  // created() {

  // }

});
</script>
