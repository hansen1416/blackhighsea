<template>
  <div class="home">
    <ChartLine
      :stock-code="stockCode"
    />
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import ChartLine from '@/components/ChartLine.vue'; // @ is an alias to /src

export default defineComponent({
  components: {
    ChartLine
  },
  computed: {
    stockCode(): string {
      return '600104';
    }
  },
  // created() {

  // }

});
</script>
