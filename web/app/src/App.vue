<template>
    <router-view />
</template>
<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
    data() {
        return {
            ws: WebSocket,
        };
    },
    created() {
        this.ws = new WebSocket("ws://127.0.0.1:4601/");

        // console.log(ws);

        this.ws.onmessage = (event) => {
            console.log(event);
        };
    },
    watch: {
        ws(newWs, oldWs) {
          console.log(newWs.readyState)
            if (newWs.readyState == 1) {
                newWs.send("123123");
            }
        },
    },
});
</script>
<style lang="scss">
html,
body {
    margin: 0;
    padding: 0;
}
</style>
