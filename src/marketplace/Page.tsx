import React, { useEffect, useState } from "react"
import AddFlatForm from "../components/newHDBForm/form";


interface Post {
  id: number | string;
  title: string;
  description: string;
  imageUrl: string;
  date: string;
  datetime:string;
  category: string;
  location: string;
  author: {
    name: string;
    telegram: string;
  };
}

  export default function HDBMarketPage() {
      const [form, setForm] = useState<boolean>(false);
      const [posts, setPosts] = useState<Post[]>([])

      const getListings = async (): Promise<void> => {
      const response = await fetch('https://vqe2yhjppn.ap-southeast-1.awsapprunner.com/api/listings', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error fetching listings:', errorData);
        alert('Error fetching listings. Please try again.');
        return;
      }
      
      const formatted_response = await response.json();
        setPosts(formatted_response);
  };
  useEffect(() => {
      getListings();
  }, []);
  return (
    <div className="pb-12 sm:pb-16 pt-8">
        {form && <AddFlatForm setForm={setForm} />}
      {!form && <div className="mx-auto max-w-7xl px-8 lg:px-4 mt-4">
        <div className="mx-auto max-w-2xl lg:max-w-5xl">
        <div className="flex flex-row justify-between">
        <h2 className="text-pretty text-4xl font-semibold tracking-tight text-white sm:text-5xl">
         Flats for sale    
        </h2>
        <button
                type="button"
                onClick={() =>setForm(true)}
                className="flex flex-row gap-x-3 items-center rounded-full mr-5 bg-white hover:bg-black hover:text-white text px-4 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 "
        >
            Create new Listing
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="white" className="size-5 bg-black rounded-full">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
            </svg>

        </button>
      </div>
          <p className="mt-2 text-lg/8 text-gray-200">Browse recently listed flats (Auto expires in 30 days)</p>
          <div className="overflow-auto scrollbar-hide max-h-[75vh] mt-8 space-y-20 lg:mt-10 lg:space-y-20">
            {posts.map((post) => (
              <article key={post.id} className="relative isolate flex flex-col gap-8 lg:flex-row">
                <div className="relative aspect-video sm:aspect-[2/1] lg:aspect-square lg:w-64 lg:shrink-0">
                  <img
                    alt=""
                    src={post.imageUrl}
                    className="absolute inset-0 size-full rounded-2xl bg-gray-50 object-cover"
                  />
                  <div className="absolute inset-0 rounded-2xl ring-1 ring-inset ring-gray-900/10" />
                </div>
                <div>
                  <div className="flex items-center gap-x-4 text-xs">
                    <p
                      className="relative z-10 rounded-full pr-3 py-1.5 font-medium text-gray-200 hover:bg-gray-100"
                    >
                      {post.date}
                    </p>
                    <p
                      className="relative z-10 rounded-full bg-gray-50 px-3 py-1.5 font-medium text-gray-600 hover:bg-gray-100"
                    >
                      {post.category}
                    </p>
                    <p
                      className="relative z-10 rounded-full bg-gray-50 px-3 py-1.5 font-medium text-gray-600 hover:bg-gray-100"
                    >
                      {post.location}
                    </p>
                  </div>
                  <div className="group relative max-w-xl">
                    <h3 className="mt-3 text-lg/6 font-semibold text-white">
                      <p>
                        <span className="absolute inset-0" />
                        {post.title}
                      </p>
                    </h3>
                    <p className="mt-5 text-sm/6 text-gray-200">{post.description}</p>
                  </div>
                  <div className="mt-6 flex border-t border-gray-900/5 pt-6">
                    <div className="relative flex items-center gap-x-4">
                      <img alt="" src='Telegram_logo.svg.webp' className="size-10 rounded-full bg-gray-50" />
                      <div className="text-sm/6">
                        <p className="font-semibold text-white">
                          <p>
                            <span className="absolute inset-0" />
                            {post.author.name}
                          </p>
                        </p>
                        <p className="text-gray-600">{post.author.telegram}</p>
                      </div>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </div>
      </div>}
    </div>
  )
}
